"""Mechanisms, as code.

Why this file exists
--------------------
The campaign's literature corpus holds 41 live causal mechanisms. Its EXPERIMENT space
held five, because every mechanism had to be hand-wired into `make_variant.build()` as an
inline `if cfg.get(...)` branch, and separately declared in `direction.MECHANISMS`, and
separately taught to the policy's key vocabulary. Nobody did that work at the rate the
readers registered mechanisms, so the corpus outran what could be run -- and the campaign
spent blocks turning knobs while twenty registered mechanisms sat unrunnable.

That gap was never a property of the mechanisms. Every one of them is a `train.py` edit,
and `train.py` is the file we are allowed to edit; upstream's rule is that architecture,
optimizer and objective are all fair game. The gap was unwritten code, and counting it as
"not buildable" turned a backlog into an imaginary constraint.

`@mechanism` closes it. One decorated function carries the edit, the diagnostic that
proves it engaged, and its companion parameters; `direction.py` reads the registry, so
registering here is sufficient to make the policy, the queue doors, the labeller and the
verdict machinery all see it. Adding the next mechanism is one function in this file.

Each entry below names the registered mechanism record and the claims it descends from,
so an edit here can be walked back to a paper on disk -- E1/E2 of the chain of evidence.
"""
from __future__ import annotations

from make_variant import mechanism


# ---------------------------------------------------------------------------
# input_pipeline
# ---------------------------------------------------------------------------

@mechanism(
    "prefetch",
    family="input_pipeline",
    diagnostic="loader_frac",
    params=(),
    doc="""Move the frozen packing loop off the main thread.

    THIS MECHANISM WAS DECLARED AND NEVER IMPLEMENTED. `direction.MECHANISMS` has listed
    `prefetch` since the campaign began, so the policy counted it toward mechanism
    coverage, offered it in the DIRECTION SPACE table and let `hyp_loader_prefetch_thread_r1`
    register against it -- while `make_variant.build()` had no branch for it, so building
    it raised VariantEditError. The mechanism record says so in its own notes ("prefetch
    is also, as of this record, not implemented in tools/make_variant.py at all") and the
    hypothesis carries a `_needs_build` field spelling out the edit. A name in a coverage
    table with no code behind it is worse than an absent mechanism: it makes the campaign
    report a direction as reachable while nothing can ever launch on it.

    Mechanism: loader_slack_prices_gpu_flops_at_zero_merged. Hypothesis:
    hyp_loader_prefetch_thread_r1 (predicts -0.003 bpb, 16x the counterbalanced resolution).

    THE DATA CONTRACT, which is the whole risk here. `make_dataloader` packs into ONE
    preallocated pinned `cpu_buffer`, issues `gpu_buffer.copy_(cpu_buffer, non_blocking=True)`,
    and yields VIEWS of that single `gpu_buffer`. Two distinct races follow, and the
    hypothesis as registered only guards the first:

      1. The consumer race. The next `next()` overwrites the buffer the trainer is still
         holding. Guarded by cloning in the producer, which the hypothesis specifies.
      2. The producer race, which it does not mention. The H2D copy is ASYNCHRONOUS from
         pinned memory, so the producer can repack `cpu_buffer` for batch k+1 while the
         copy for batch k has not yet executed -- silently feeding the model a blend of
         two batches. In the single-threaded baseline this is safe only by accident: the
         main thread does a full forward/backward between `next()` calls. A producer
         thread running two batches ahead removes that accident.

    Both are closed here. The producer runs on its OWN cuda stream, so the generator's H2D
    lands there, and after cloning it blocks on an event recorded on that stream. The event
    covers the copy and the clone and NOTHING ELSE -- in particular not the model kernels
    on the default stream, so waiting on it does not serialize the producer against
    training, which a plain `torch.cuda.synchronize()` would. Once it returns, `cpu_buffer`
    is provably reusable. The consumer calls `record_stream` on the handed-off tensors
    because they were allocated on the producer's stream and are consumed on the default
    one; without it the caching allocator may recycle those blocks while training reads
    them, which is the same corruption by a different route.

    Data ORDER, epoch accounting and `prepare.py` itself are untouched: one thread consumes
    the one generator in the one order. The falsifiers require final_epoch, tokens_per_step
    and grad_accum to match the yoked control, and a run that violates them is VOIDED, not
    scored -- a throughput win bought with a data-contract violation is not a win.""",
)
def _prefetch(s, cfg, sub):
    depth = int(cfg["prefetch"])
    s = sub(s, """train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train")
x, y, epoch = next(train_loader)  # prefetch first batch""",
f"""train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train")
import threading as _pf_threading, queue as _pf_queue
PREFETCH_DEPTH = {depth}
_pf_q = _pf_queue.Queue(maxsize=PREFETCH_DEPTH)
_pf_stream = torch.cuda.Stream()
_pf_stat = [0, 0]          # [gets that found the queue empty, total gets]
_pf_err = []

def _pf_worker():
    # The generator's own H2D copy is issued on THIS stream because the stream context is
    # per-thread and the generator body runs inside this call.
    try:
        with torch.cuda.stream(_pf_stream):
            while True:
                _bx, _by, _bep = next(train_loader)
                _cx, _cy = _bx.clone(), _by.clone()
                _ev = torch.cuda.Event()
                _ev.record(_pf_stream)
                # Blocks until the copy AND the clone have executed, so the shared pinned
                # cpu_buffer is safe for this thread to repack. Scoped to the loader
                # stream, so it does not wait on training kernels.
                _ev.synchronize()
                _pf_q.put((_cx, _cy, _bep))
    except BaseException as _e:                  # a dead producer must not hang the run
        _pf_err.append(_e)
        _pf_q.put(None)

_pf_threading.Thread(target=_pf_worker, daemon=True).start()

def _pf_next():
    _pf_stat[1] += 1
    if _pf_q.empty():
        _pf_stat[0] += 1
    _item = _pf_q.get()
    if _item is None:
        raise RuntimeError(f"prefetch producer died: {{_pf_err[0]!r}}")
    _bx, _by, _bep = _item
    _cs = torch.cuda.current_stream()
    _bx.record_stream(_cs); _by.record_stream(_cs)
    return _bx, _by, _bep

x, y, epoch = _pf_next()  # prefetch first batch""")
    # The in-loop site as the shared instrumentation block has already rewritten it.
    s = sub(s, """        x, y, epoch = next(train_loader)
        _tc = time.perf_counter()""",
            """        x, y, epoch = _pf_next()
        _tc = time.perf_counter()""")
    obs = ["""print(f"queue_starve_frac: {_pf_stat[0]/max(_pf_stat[1],1):.6f}")
print(f"prefetch_depth:    {PREFETCH_DEPTH}")"""]
    return s, obs


# ---------------------------------------------------------------------------
# ve_placement
# ---------------------------------------------------------------------------

@mechanism(
    "vefreeze",
    family="ve_placement",
    diagnostic="ve_trainable_params",
    params=(),
    doc="""Freeze the value-embedding tables at their random init, keeping the gated
    pathway that injects them fully trainable.

    This is the PATHWAY-vs-CONTENT control, and it is the single most important arm the
    campaign is missing. Value embeddings are its most reliable winning family, and the
    queued n-gram arm is another table-shaped lookup in the same family. Both are assumed
    to work because the TABLE LEARNS SOMETHING. `embA_engram_image_pathway_not_memory`
    reports the opposite in autoregressive image generation: the gain came from the gated
    additive side-path, and replacing the learned table with FROZEN RANDOM NOISE cost
    almost nothing. If that transfers, then every value-embedding result this campaign has
    banked is a result about an extra input-dependent pathway into V, not about memory --
    and the whole n-gram direction is aimed at the wrong quantity.

    Mechanism: ve_pathway_not_table_content. Claims: embA_engram_image_pathway_not_memory
    (opposes, transfer 1 -- vision, so this arm is exactly the transfer test), with
    embA_engram_beats_raw_gpt_bpb and value_embedding_density as the supporting side.

    The edit removes the value-embedding parameter group from the optimizer and clears
    requires_grad, so the tables keep their `uniform_(-s, s)` init for the whole run while
    `ve_gate` still learns. FLOPs are unchanged; the optimizer does strictly less work.
    Note the table is NOT zero-init, so freezing leaves a genuine random code -- this
    ablates learned content, not the pathway. Zero-init would have made it an ablation of
    the pathway itself, which is the arm this one exists to be distinguished from.""",
)
def _vefreeze(s, cfg, sub):
    s = sub(s, """        value_embeds_params = list(self.value_embeds.parameters())""",
            """        value_embeds_params = list(self.value_embeds.parameters())
        for _p in value_embeds_params:
            _p.requires_grad_(False)""")
    s = sub(s, """            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
""", "")
    # The trainer asserts every parameter is accounted for by a group; frozen tables are
    # accounted for by being frozen, so the assert is retargeted rather than removed.
    s = sub(s, """        assert len(list(self.parameters())) == (len(matrix_params) + len(embedding_params) +
            len(lm_head_params) + len(value_embeds_params) + len(resid_params) + len(x0_params))""",
            """        assert len(list(self.parameters())) == (len(matrix_params) + len(embedding_params) +
            len(lm_head_params) + len(value_embeds_params) + len(resid_params) + len(x0_params))
        assert not any(p.requires_grad for p in value_embeds_params)""")
    # ve_emb_rms_final is NOT printed here: the shared telemetry block already emits it
    # from the model. Printing it twice made this the only variant with a duplicated key,
    # and host/dispatch.py parses with `m[key] = value`, so which number survived was
    # decided by print order -- reorder these lines and the record silently becomes the
    # RMS of a different tensor.
    obs = ["""_vem = (model._orig_mod if hasattr(model,'_orig_mod') else model)
_vetr = sum(p.numel() for p in _vem.value_embeds.parameters() if p.requires_grad)
print(f"ve_trainable_params: {_vetr}")"""]
    return s, obs


@mechanism(
    "embwd",
    family="ve_placement",
    diagnostic="emb_wnorm_final",
    params=(),
    # embwd=1 is an ABLATION, not a regularisation: the mechanism's own validator
    # computes that it shrinks the identity tables to ~1e-244 of their initial norm over
    # this budget and refuses. 1e-3 sits an order of magnitude under that validator's
    # stated ceiling, so the screen measures regularisation rather than deletion.
    screen_value=1e-3,
    doc="""Apply weight decay to the token-identity tables -- wte, the value embeddings and
    lm_head -- which currently receive none.

    Mechanism: decay_on_the_token_identity_tables. Its mediator is the weight-norm
    trajectory of the 41.67% of parameters that get no shrinkage at all: all three AdamW
    groups are declared `weight_decay=0.0`, while the Muon matrix groups take the decayed
    `WEIGHT_DECAY * (1 - progress)` schedule. So the model's single largest parameter block
    is the one block nothing regularises, at a budget short enough that these tables see
    each token id only a handful of times.

    Zero added FLOPs -- it is one term in an update that already runs. The value is the
    constant decay applied to those three groups.""",
)
def _embwd(s, cfg, sub):
    from make_variant import VariantEditError
    wd = float(cfg["embwd"])
    # SCALE THIS KNOB AGAINST ITS OWN LEARNING RATE, or the obvious values annihilate the
    # tables. AdamW here is DECOUPLED (p.mul_(1 - lr_t * wd_t)), and the embedding and
    # value-embedding groups run at lr = 0.6 * (512/768)**-0.5 = 0.7348 -- 18x Muon's
    # 0.04. So the same written number means something wildly different in the two
    # places, and "0.2, like WEIGHT_DECAY" or even "0.01, a small first value" would end
    # the run having erased the tables: 0.01 leaves ~2% of the initial norm, 0.05 leaves
    # 3e-9. That arm would come back as a clean valid-negative on "decay the identity
    # tables" when what it actually tested was deleting them.
    _lr_eff = 0.6 * (512 / 768) ** -0.5 * 0.75      # 0.75 = mean lr multiplier over the run
    _steps = 700                                     # the campaign's usual step count
    _mult = (1.0 - _lr_eff * wd) ** _steps if _lr_eff * wd < 1 else 0.0
    if _mult < 0.5:
        raise VariantEditError(
            f"embwd={wd} would shrink the identity tables to {_mult:.3g} of their initial "
            f"norm over ~{_steps} steps (decoupled decay at lr={_lr_eff:.3f}). That is an "
            f"ablation, not a regularisation. Keep the predicted multiplier above 0.5: "
            f"embwd <= {(1 - 0.5 ** (1 / _steps)) / _lr_eff:.2e}.")
    for grp in ("lm_head_params", "embedding_params", "value_embeds_params"):
        lr = {"lm_head_params": "unembedding_lr", "embedding_params": "embedding_lr",
              "value_embeds_params": "embedding_lr"}[grp]
        s = sub(s, f"""            dict(kind='adamw', params={grp}, lr={lr} * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),""",
                f"""            dict(kind='adamw', params={grp}, lr={lr} * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay={wd}),""")
    obs = [f"""_ewm = (model._orig_mod if hasattr(model,'_orig_mod') else model)
_ewp = [_ewm.transformer.wte.weight, _ewm.lm_head.weight] + list(_ewm.value_embeds.parameters())
print(f"emb_wnorm_final:  {{sum(p.detach().float().pow(2).sum().item() for p in _ewp)**0.5:.6f}}")
print("emb_wd_applied:   {wd}")"""]
    return s, obs


# ---------------------------------------------------------------------------
# signal_path
# ---------------------------------------------------------------------------

@mechanism(
    "periln",
    family="signal_path",
    diagnostic="branch_stream_ratio_mean",
    params=(),
    doc="""Peri-LN: normalise each sub-module's OUTPUT before it is added to the residual
    stream, in addition to the existing normalisation of its input.

    Mechanism: periln_branch_output_norm. Its mediator is the branch-to-stream amplitude
    ratio RMS(sub-module output) / RMS(residual stream at that depth). The baseline is
    Pre-LN: `x = x + attn(norm(x))`, so the branch output enters the stream at whatever
    scale training happens to give it, and that scale compounds with depth through
    `resid_lambdas`. Peri-LN pins it, which is what makes the ratio a controlled quantity
    rather than an emergent one.

    Nearly zero FLOPs -- two extra RMS norms per block over tensors that are already
    resident. Note `c_proj` is zero-init in both branches, so the branch output is exactly
    zero at step 0; `F.rms_norm` with its default eps returns zero there rather than
    dividing by zero, so the init path is unchanged.""",
)
def _periln(s, cfg, sub):
    s = sub(s, """        x = x + self.attn(norm(x), ve, cos_sin, window_size)
        x = x + self.mlp(norm(x))""",
            """        x = x + norm(self.attn(norm(x), ve, cos_sin, window_size))
        x = x + norm(self.mlp(norm(x)))""")
    # NOTE: no flag is set for the shared probe. branch_out_rms_absdev was removed as an
    # arithmetic identity (see make_variant.py); periln has no diagnostic that can fail
    # and must not be queued until it does.
    # The diagnostic (branch_stream_ratio_mean) is emitted by the SHARED telemetry block
    # in make_variant.py, unconditionally, so the CONTROL emits it too. A diagnostic only
    # the treatment prints cannot be surprising -- there is no distribution to compare
    # against, and "the number exists" becomes the whole test.
    obs = []
    return s, obs


@mechanism(
    "vnorm",
    family="signal_path",
    diagnostic="branch_stream_ratio_mean",
    params=(),
    doc="""HybridNorm's QKV-Norm half: extend the existing QK normalisation to the VALUE
    path, so all three attention inputs are normalised rather than two.

    Claim: nrmC_hybridnorm_qkv_post (2503.04598) -- QKV-Norm in attention combined with
    Post-Norm in the FFN beat Pre-Norm, Post-Norm and every other hybrid tested in a
    controlled 550M/400B-token ablation. Zero added FLOPs; it is a placement change over
    tensors already in registers. Transfer is the open question and is scored 1/4: the
    evidence is at roughly 10^5 times our token budget with no seeds reported.

    Split deliberately from the FFN half. The paper's result is for the COMBINATION, but
    the two halves have very different risk here: normalising V is local to attention and
    cannot change the residual stream's scale, whereas moving the FFN norm post-residual
    interacts with `resid_lambdas`, `x0_lambdas` and the zero-init `c_proj`. Testing the
    cheap safe half alone first is the campaign's own rule -- evaluate new structure before
    entangling it. `ffnpost` is the other half.

    The norm is applied AFTER the value-embedding mix, so it also pins the scale the VE
    pathway injects at; that makes this arm partially confounded with the ve_placement
    family, which the verdict must state rather than the arm hide.""",
)
def _vnorm(s, cfg, sub):
    s = sub(s, """        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)""",
            """        v = norm(v)
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)""")
    # The diagnostic (branch_stream_ratio_mean) is emitted by the SHARED telemetry block
    # in make_variant.py, unconditionally, so the CONTROL emits it too. A diagnostic only
    # the treatment prints cannot be surprising -- there is no distribution to compare
    # against, and "the number exists" becomes the whole test.
    obs = []
    return s, obs


@mechanism(
    "ffnpost",
    family="signal_path",
    diagnostic="branch_stream_ratio_mean",
    params=(),
    doc="""HybridNorm's Post-Norm half: normalise the residual stream AFTER the FFN branch
    is added, instead of normalising only the FFN's input.

    Claim: nrmC_hybridnorm_qkv_post. This is the riskier half of that paper's recipe here
    (see `vnorm`): it rescales the stream itself, which `resid_lambdas` and `x0_lambdas`
    then multiply, so an interaction with the campaign's tuned residual scalars is
    expected rather than surprising. That is exactly why it is a separate arm.

    Zero added FLOPs: the FFN's input norm is removed and one stream norm is added.""",
)
def _ffnpost(s, cfg, sub):
    s = sub(s, """        x = x + self.mlp(norm(x))""",
            """        x = norm(x + self.mlp(x))""")
    # The diagnostic (branch_stream_ratio_mean) is emitted by the SHARED telemetry block
    # in make_variant.py, unconditionally, so the CONTROL emits it too. A diagnostic only
    # the treatment prints cannot be surprising -- there is no distribution to compare
    # against, and "the number exists" becomes the whole test.
    obs = []
    return s, obs


# ---------------------------------------------------------------------------
# attention
# ---------------------------------------------------------------------------

@mechanism(
    "winsched",
    family="attention",
    diagnostic="winsched_start_window",
    params=("winsched_frac",),
    doc="""SkyLadder: expand the SSSL short attention window from a small value up to the
    platform's short window over the first part of the run, instead of holding it fixed.

    Claims: attB_skyladder_window_schedule (2503.15450) -- short-to-long window scheduling
    improved fixed-budget accuracy by 3.7 points AND cut wall-clock training time 13-22%
    against a constant-window baseline; attB_skyladder_schedule_hparams -- at the paper's
    smallest scale (120M) a tiny initial window expanding over roughly 64% of the run beat
    both a larger start and a faster expansion.

    Why this arm ranks above the rest of the incoming literature. The campaign's single
    best-established lever is `swdiv`: shrinking the SSSL short span won four consecutive
    times and produced the standing best run, because removing attention FLOPs buys steps
    and local recall is covered elsewhere. Every one of those experiments varied a CONSTANT
    span. SkyLadder says the constant is the wrong object -- that the span should be small
    exactly when the model cannot yet use long-range structure, and grow later. It is a new
    AXIS on a confirmed lever rather than another point on a saturated one, and it is
    FLOP-REDUCING early, which is the shape of every win this campaign has banked. It is
    also the only paper in this round reporting a wall-clock number, which is the currency
    of a 300-second budget.

    The window is recomputed per step from `progress`, so it costs one Python comparison
    per step and no GPU work. `winsched` is the starting short window in tokens;
    `winsched_frac` is the fraction of the run over which it reaches the platform value
    (default 0.64, the paper's own best at its smallest scale).

    THE COMPILE HAZARD, which decides whether this arm measures anything. `window_size` is
    an argument to `fa3.flash_attn_func` inside a `torch.compile`d module. A value that
    changes every step would retrigger compilation every step and the run would measure the
    compiler, not the mechanism. The schedule is therefore QUANTISED to powers of two, so
    the window takes a handful of distinct values over the run and each triggers at most
    one recompile -- and the count of distinct values is printed, so the cost is visible in
    the record rather than inferred.""",
)
def _winsched(s, cfg, sub):
    from make_variant import VariantEditError
    start = int(cfg["winsched"])
    frac = float(cfg.get("winsched_frac") or 0.64)
    # A schedule that starts at the span it is ramping TO cannot move. With swdiv=16 the
    # short span is already 128, so {swdiv:16, winsched:128} builds byte-DIFFERENT from
    # the control -- it carries the whole scheduler -- and then holds the window fixed
    # for the entire run. It would pass the byte-identical queue guard and burn a 300s
    # slot as a real attention-axis run measuring nothing. `ns` refuses its own no-op at
    # build time for exactly this reason; this now does too.
    _seq, _swdiv = 2048, int(cfg.get("swdiv") or 2)
    _short = _seq // _swdiv
    if start >= _short:
        raise VariantEditError(
            f"winsched={start} is a NO-OP: the short span under swdiv={_swdiv} is already "
            f"{_short}, so the schedule starts at its own target and never moves. Pick a "
            f"start below {_short}, or change swdiv.")
    s = sub(s, """    def _compute_window_sizes(self, config):""",
            f"""    def set_short_window(self, w):
        # Rebuild the per-layer window list with a new SHORT span; long layers and the
        # forced-long final layer are untouched, so the SSSL pattern is preserved.
        cfg = self.config
        pattern = cfg.window_pattern.upper()
        long_window = cfg.sequence_len
        sizes = []
        for layer_idx in range(cfg.n_layer):
            char = pattern[layer_idx % len(pattern)]
            sizes.append((long_window, 0) if char == "L" else (int(w), 0))
        sizes[-1] = (long_window, 0)
        self.window_sizes = sizes

    def _compute_window_sizes(self, config):""")
    s = sub(s, """    progress = min(total_training_time / TIME_BUDGET, 1.0)
    lrm = get_lr_multiplier(progress)""",
f"""    progress = min(total_training_time / TIME_BUDGET, 1.0)
    _ws_target = _WS_FULL if progress >= {frac} else int(2 ** round(
        math.log2({start}) + (math.log2(_WS_FULL) - math.log2({start})) * (progress / {frac})))
    _ws_target = max({start}, min(_WS_FULL, _ws_target))
    if _ws_target != _ws_cur[0]:
        _ws_cur[0] = _ws_target
        _ws_seen.add(_ws_target)
        (model._orig_mod if hasattr(model, '_orig_mod') else model).set_short_window(_ws_target)
    lrm = get_lr_multiplier(progress)""")
    s = sub(s, """train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train")""",
f"""# The SHORT span is the minimum over the per-layer windows, NOT window_sizes[0].
# Position 0 is whatever the pattern's first character says: under any pattern starting
# with 'L' (e.g. "LSSS") it is the FULL sequence length, so the ramp would have
# terminated by setting every short layer to full context -- strictly MORE attention
# FLOPs than the control, under a mechanism whose entire thesis is FLOP reduction, and
# `win` is an untouched top-priority axis so win x winsched is a live proposal.
_WS_FULL = min(w[0] for w in (model._orig_mod if hasattr(model, '_orig_mod') else model).window_sizes)
_ws_cur = [_WS_FULL]
_ws_seen = set()
(model._orig_mod if hasattr(model, '_orig_mod') else model).set_short_window({start})
_ws_cur[0] = {start}
_ws_seen.add({start})
train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train")""")
    obs = [f"""print(f"winsched_start_window: {start}")
print(f"winsched_full_window:  {{_WS_FULL}}")
print(f"winsched_distinct:     {{len(_ws_seen)}}")
print(f"winsched_final_window: {{_ws_cur[0]}}")"""]
    return s, obs


@mechanism(
    "ropefrac",
    family="attention",
    diagnostic="rope_rotated_dims",
    params=(),
    doc="""Fractional RoPE: apply the rotary rotation to only the first `ropefrac` fraction
    of each head's channels, leaving the remainder unrotated (NoPE).

    Claim: attB_fractional_rope_10pct (2603.11611) -- rotating roughly 10% of head-dim
    channels matched full-RoPE perplexity (23.25 vs 22.99) across three architectures,
    while 0-4% was much worse. A systematic sweep, so the shape of the curve is known and
    not just its endpoint: there is a floor, and above it the extra rotated channels buy
    nothing.

    Paired caution from the same paper, registered separately as
    attB_partial_rope_nope_instability: driving this to full NoPE without QK-norm caused
    catastrophic divergence (perplexity 340,933). We run QK-norm on the platform, which the
    claim identifies as the thing that prevents it -- so this arm is safe HERE for a stated
    reason, and it must never be crossed with `noqknorm`.

    Fewer FLOPs and less memory traffic, zero new parameters: the rotation is a slice
    rather than a full elementwise pass. Same family as the confirmed `swdiv` win --
    removing attention work at fixed budget.""",
)
def _ropefrac(s, cfg, sub):
    from make_variant import VariantEditError
    fr = float(cfg["ropefrac"])
    # PROSE IS NOT A GUARD. The docstring above says this "must never be crossed with
    # noqknorm", citing attB_partial_rope_nope_instability -- reducing positional
    # information without QK-norm diverged to perplexity 340,933. That sentence stopped
    # nothing: the cross built cleanly, no lesson blocked the key pair, and neither queue
    # door checks pairwise incompatibility. The only stated reason this arm is safe here
    # is that the platform runs QK-norm, so removing QK-norm removes the reason.
    # `is not None`, matching how the noqknorm branch itself tests. A truthiness test let
    # noqknorm:0 and noqknorm:false through -- both APPLY the edit (the legacy branch
    # fires on `is not None`) while reading to a human as "QK-norm on", so the evasion
    # looked like the safe configuration.
    if cfg.get("noqknorm") is not None:
        raise VariantEditError(
            "ropefrac x noqknorm is refused: fractional RoPE is safe here only BECAUSE "
            "the platform runs QK-norm (attB_partial_rope_nope_instability -- ppl 340,933 "
            "without it). Test them separately.")
    # The baseline uses the SPLIT-HALF convention: channel i is paired with channel i+d,
    # where d = head_dim/2, and cos/sin are indexed by PAIR. Slicing a contiguous leading
    # block of channels would therefore re-pair them against different partners and leave
    # a "tail" drawn from both halves -- a different edit from the one claimed. This
    # rotates a fraction of the PAIRS and passes the rest through, so every rotated channel
    # keeps the partner it has in the control.
    s = sub(s, """def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)""",
f"""ROPE_FRAC = {fr}

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    r = max(1, min(d, int(d * ROPE_FRAC)))    # number of pairs that get rotated
    x1, x2 = x[..., :d], x[..., d:]
    if r < d:
        a1, b1 = x1[..., :r], x1[..., r:]
        a2, b2 = x2[..., :r], x2[..., r:]
        c, sn = cos[..., :r], sin[..., :r]
        y1 = torch.cat([a1 * c + a2 * sn, b1], -1)
        y2 = torch.cat([a1 * (-sn) + a2 * c, b2], -1)
        return torch.cat([y1, y2], 3)
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)""")
    obs = [f"""_rp = max(1, min(HEAD_DIM // 2, int((HEAD_DIM // 2) * {fr})))
print(f"rope_rotated_dims: {{2 * _rp}}")
print(f"rope_head_dim:     {{HEAD_DIM}}")"""]
    return s, obs
