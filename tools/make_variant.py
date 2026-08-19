#!/usr/bin/env python3
"""Generate one train.py variant from a config dict. Only train.py is ever edited."""
import json
import pathlib
import re
import sys

BASE = pathlib.Path(__file__).resolve().parent.parent / "baseline" / "train.py"

class VariantEditError(RuntimeError):
    """A requested edit did not apply. Raised instead of silently producing a variant
    that is byte-identical to the baseline.

    This is THE bug class that corrupted the v3 campaign. `str.replace` on an absent
    target returns the string unchanged and raises nothing, so a mechanism could be
    "tested" while its code was never inserted. It happened at least three times:
    the `ema` activation print targeted a line that does not exist in build() order,
    so all three ema arms ran with no engagement observable; orphan recovery wrote
    cfg:{} so runs counted against no axis; and truthiness tests skipped zero-valued
    knobs. Every substitution below now asserts it changed the source.
    """


def sub(s: str, old: str, new: str, what: str = "") -> str:
    """Replace exactly once and prove the target was present.

    An ABSENT target is always a bug: the caller believed it was editing something.
    An identical replacement (old == new, e.g. dbs=128 written over 128) is legitimate
    and passes through -- that is a config that happens to equal the baseline, not a
    failed edit.
    """
    if old not in s:
        raise VariantEditError(f"edit target not found ({what or old[:70]!r})")
    return s.replace(old, new, 1)


def variant_id(src: str) -> str:
    """Content-addressed variant filename.

    Must be a STABLE digest. `hash()` was used here and is randomized per process by
    PYTHONHASHSEED, so byte-identical variant source produced a different filename in
    every process -- defeating dedup and making the mapping from a queue entry to its
    source irreproducible after the fact.
    """
    import hashlib
    return hashlib.sha256(src.encode()).hexdigest()[:12] + ".py"


def build(cfg: dict) -> str:
    s = BASE.read_text()
    obs = []   # activation-observable print lines, injected after the telemetry block

    # --- instrumentation: time decomposition + richer telemetry (no math change) ---
    s = sub(s, "t_start_training = time.time()",
                  "t_start_training = time.time()\n_t_fwdbwd=0.0;_t_loader=0.0;_t_opt=0.0;_t_wait=0.0;_dts=[]")
    s = sub(s, """    for micro_step in range(grad_accum_steps):
        with autocast_ctx:
            loss = model(x, y)
        train_loss = loss.detach()
        loss = loss / grad_accum_steps
        loss.backward()
        x, y, epoch = next(train_loader)
""", """    for micro_step in range(grad_accum_steps):
        _ta = time.perf_counter()
        with autocast_ctx:
            loss = model(x, y)
        train_loss = loss.detach()
        loss = loss / grad_accum_steps
        loss.backward()
        _tb = time.perf_counter()
        x, y, epoch = next(train_loader)
        _tc = time.perf_counter()
        if step > 10:
            _t_fwdbwd += _tb-_ta; _t_loader += _tc-_tb
""")
    s = sub(s, """    if step > 10:
        total_training_time += dt
""", """    if step > 10:
        total_training_time += dt; _dts.append(dt)
""")

    # --- second-moment telemetry, emitted by EVERY variant including the control ------
    # `second_momentum_buffer` holds the NorMuon second moment. Where that block sits
    # relative to the polar iteration decides what the buffer MEANS: the second moment of
    # the orthogonalized update (baseline) or of the raw momentum (precond="pre"). Those
    # live on structurally different scales -- an orthogonalized matrix has entries of
    # order 1/sqrt(max(A,B)) by construction -- so the same statistic printed by both arms
    # is a real engagement test. Printed only in the treatment it would be a number with
    # nothing to be wrong against.
    #
    # secmom_clamp_frac is the failure mode that rms alone hides: the buffer is zero-init
    # with 1-beta2=0.05 and no bias correction, so `clamp_min(1e-10)` can bind. When it
    # does, the diagonal collapses to a constant, the polar iteration's own Frobenius
    # normalization cancels it exactly, and the arm silently degenerates to plain Muon
    # minus NorMuon -- a "did not engage" that looks like a clean null.
    #
    # All of it runs after the charged clock stops, so it cannot move val_bpb or steps.
    obs.append("""_sm = [st["second_momentum_buffer"] for st in optimizer.state.values()
       if "second_momentum_buffer" in st]
if _sm:
    _v = torch.cat([b.float().flatten() for b in _sm])
    print(f"secmom_rms:          {_v.mean().sqrt().item():.8f}")
    print(f"secmom_max:          {_v.max().sqrt().item():.8f}")
    print(f"secmom_clamp_frac:   {(_v <= 1e-10).float().mean().item():.8f}")
    print(f"secmom_nbuf:         {len(_sm)}")
    # Self-contained engagement number. An orthogonalized matrix has entries of order
    # 1/sqrt(max(A,B)) BY CONSTRUCTION, so if the buffer is tracking the polar output its
    # rms sits at that reference and this ratio is ~1; if it is tracking the raw momentum
    # the ratio departs from 1 by orders of magnitude. Emitting the RATIO rather than the
    # raw rms is what lets a single run's activation be checked by a machine rule, instead
    # of a cross-arm comparison that lives only in prose.
    _ref = torch.tensor([ (1.0/max(b.shape[-2], b.shape[-1])**0.5) for b in _sm ]).mean()
    print(f"secmom_ortho_ratio:  {(_v.mean().sqrt()/_ref).item():.8f}")""")

    # --- evaluator pinned to the baseline batch so the metric stays comparable ---
    s = sub(s, "val_bpb = evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE)",
                  "val_bpb = evaluate_bpb(model, tokenizer, 128)")

    # --- knobs ---
    s = sub(s, "DEVICE_BATCH_SIZE = 128", f"DEVICE_BATCH_SIZE = {cfg['dbs']}")
    s = sub(s, "TOTAL_BATCH_SIZE = 2**19", f"TOTAL_BATCH_SIZE = 2**{cfg['tbs']}")
    if cfg.get("mlp", 4) != 4:
        m = cfg["mlp"]
        s = sub(s, "nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)",
                      f"nn.Linear(config.n_embd, {m} * config.n_embd, bias=False)")
        s = sub(s, "nn.Linear(4 * config.n_embd, config.n_embd, bias=False)",
                      f"nn.Linear({m} * config.n_embd, config.n_embd, bias=False)")
    if cfg.get("ve", 2) != 2:
        s = sub(s, "    return layer_idx % 2 == (n_layer - 1) % 2",
                      f"    return (n_layer - 1 - layer_idx) % {cfg['ve']} == 0")
    if cfg.get("swdiv", 2) != 2:
        s = sub(s, "short_window = long_window // 2",
                      f"short_window = long_window // {cfg['swdiv']}")
    s = sub(s, 'WINDOW_PATTERN = "SSSL"', f'WINDOW_PATTERN = "{cfg.get("win","SSSL")}"')

    # explicit depth/width: bypass ASPECT_RATIO so depth and width move independently
    s = sub(s, """    return GPTConfig(
        sequence_len=MAX_SEQ_LEN, vocab_size=vocab_size,
        n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
        window_pattern=WINDOW_PATTERN,
    )""", f"""    model_dim = {cfg['dim']}
    num_heads = model_dim // HEAD_DIM
    return GPTConfig(
        sequence_len=MAX_SEQ_LEN, vocab_size=vocab_size,
        n_layer={cfg['depth']}, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
        window_pattern=WINDOW_PATTERN,
    )""")


    # ---------------- MECHANISMS (new causal structure, not new constants) ----------------

    # M1 objective/supervision density: auxiliary t+2 head, weight annealed to 0.
    # Gated on self.training so evaluate_bpb's path is bit-identical.
    if cfg.get("mtp") is not None:
        s = sub(s, "        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)",
"""        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.mtp_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)""")
        s = sub(s, "            torch.nn.init.zeros_(block.mlp.c_proj.weight)",
"""            torch.nn.init.zeros_(block.mlp.c_proj.weight)
        torch.nn.init.zeros_(self.mtp_proj.weight)""")
        s = sub(s, """        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                   ignore_index=-1, reduction=reduction)
            return loss""",
"""        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                   ignore_index=-1, reduction=reduction)
            if self.training and reduction == 'mean':
                t2 = torch.cat([targets[:, 1:], torch.full_like(targets[:, :1], -1)], dim=1)
                l2 = self.lm_head(norm(x + self.mtp_proj(x)))
                l2 = softcap * torch.tanh(l2.float() / softcap)
                aux = F.cross_entropy(l2.view(-1, l2.size(-1)), t2.reshape(-1),
                                      ignore_index=-1, reduction='mean')
                loss = loss + MTP_W * aux
            return loss""")
        s = sub(s, "        lm_head_params = list(self.lm_head.parameters())",
                      "        lm_head_params = list(self.lm_head.parameters()) + list(self.mtp_proj.parameters())")
        s = re.sub(r"        assert len\(list\(self\.parameters\(\)\)\) == \(len\(matrix_params\)[^)]*\)[^)]*\)\)\n", "", s)
        s = sub(s, "HEAD_DIM = 128",
                      f"HEAD_DIM = 128\nMTP_W = {cfg['mtp']}\n_mtp_hist = []")
        s = sub(s, "                loss = loss + MTP_W * aux",
"""                _mtp_hist.append(aux.detach())
                loss = loss + MTP_W * aux""")
        obs.append("""if _mtp_hist:
    _h = [float(v) for v in _mtp_hist]
    _first = sum(_h[:100])/max(len(_h[:100]),1); _last = sum(_h[-100:])/max(len(_h[-100:]), 1)
    print(f"mtp_aux_loss_final:  {_last:.6f}")
    print(f"mtp_aux_loss_drop:   {_first-_last:.6f}")
    print(f"mtp_grad_norm_frac:  {MTP_W*_last/max(_last+float(train_loss),1e-9):.6f}")""")

    # M2 parameterization/signal path: learned cross-depth (U-net) skips, init 0 => starts
    # bit-identical to baseline, so any movement off zero proves the optimizer used the path.
    if cfg.get("unet") is not None:
        s = sub(s, "        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))",
"""        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        self.skip_lambdas = nn.Parameter(torch.zeros(config.n_layer // 2))""")
        s = sub(s, """        x = self.transformer.wte(idx)
        x = norm(x)
        x0 = x
        for i, block in enumerate(self.transformer.h):""",
"""        x = self.transformer.wte(idx)
        x = norm(x)
        x0 = x
        half = self.config.n_layer // 2
        skips = []
        for i, block in enumerate(self.transformer.h):
            if i >= self.config.n_layer - half:
                x = x + self.skip_lambdas[i - (self.config.n_layer - half)] * skips[self.config.n_layer - 1 - i]""")
        s = sub(s, """            x = block(x, ve, cos_sin, self.window_sizes[i])
        x = norm(x)""",
"""            x = block(x, ve, cos_sin, self.window_sizes[i])
            if i < half:
                skips.append(x)
        x = norm(x)""")
        s = sub(s, "        x0_params = [self.x0_lambdas]",
                      "        x0_params = [self.x0_lambdas, self.skip_lambdas]")
        obs.append("""_sl = (model._orig_mod if hasattr(model,'_orig_mod') else model).skip_lambdas.detach().abs()
print(f"skip_lambda_absmean: {_sl.mean().item():.6f}")
print(f"skip_lambda_absmax:  {_sl.max().item():.6f}")""")
        s = sub(s, """        assert len(list(self.parameters())) == (len(matrix_params) + len(embedding_params) +
            len(lm_head_params) + len(value_embeds_params) + len(resid_params) + len(x0_params))""", "")

    # M3 temporal dynamics: EMA of weights, evaluated instead of the last iterate.
    # Eval happens AFTER the charged clock stops, so the swap itself is free.
    if cfg.get("ema") is not None:
        # LAWA: average only the TAIL of training. Seeding from the random init and
        # lerping at 1e-3 leaves the average near initialisation -> evaluating noise
        # (measured: 2.88 bpb). Start the average late, and only use it if it is warm.
        s = sub(s, "t_start_training = time.time()",
"""t_start_training = time.time()
_ema = None; _ema_n = 0
_EMA_D = """ + str(cfg["ema"]) + """
_EMA_START = """ + str(cfg.get("ema_start", 0.6)))
        s = sub(s, "    step += 1",
"""    if progress >= _EMA_START and step % 4 == 0:
        with torch.no_grad():
            if _ema is None:
                _ema = {k: v.detach().float().clone() for k, v in model.named_parameters()}
            else:
                for _k, _v in model.named_parameters():
                    _ema[_k].lerp_(_v.detach().float(), 1 - _EMA_D)
            _ema_n += 1
    step += 1""")
        s = sub(s, "model.eval()",
"""if _ema is not None and _ema_n >= 10:
    with torch.no_grad():
        for _k, _v in model.named_parameters():
            _v.copy_(_ema[_k].to(_v.dtype))
model.eval()""")
        obs.append("""print(f"ema_updates:       {_ema_n}")""")

    # M4 gradient geometry: the Contrarian's discriminator -- constant Muon momentum.
    if cfg.get("mu_const") is not None:
        # The target here read `min(step / 300)` while baseline/train.py:535 reads
        # `min(step / 300, 1)`. sub() raised VariantEditError on EVERY mu_const build, so
        # this axis was unlaunchable while direction.py kept ranking it "UNEXPLORED (top
        # priority)" -- the policy was steering rounds at a knob that could not be run.
        s = sub(s, """def get_muon_momentum(step):
    frac = min(step / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95""",
f"""def get_muon_momentum(step):
    return {cfg['mu_const']}""")

    # M5 gradient geometry: orthogonalization fidelity (Newton-Schulz iteration count).
    if cfg.get("ns") is not None:
        # L007_ns_above_five_is_a_noop. polar_express_coeffs is a FIXED-LENGTH list and
        # the iteration slices it `[:ns_steps]`, so any ns at or above its length runs the
        # same iterations as the control while producing a different source hash -- a
        # variant that passes the byte-identical guard, burns a 300s slot, and is counted
        # by the policy as a real run against the ns axis. Refuse it at build time.
        # This check is what makes L007's mitigation TRUE; the lesson asserted it before
        # it existed, which an independent audit caught by simply building ns=8.
        import re as _re
        _m = _re.search(r"polar_express_coeffs\s*=\s*\[(.*?)\n\]", s, _re.S)
        _n = _m.group(1).count("(") if _m else 0
        if _n and int(cfg["ns"]) >= _n:
            raise VariantEditError(
                f"ns={cfg['ns']} is a NO-OP: polar_express_coeffs has {_n} entries and the "
                f"loop slices [:ns_steps], so this runs the same {_n} iterations as the "
                f"control while changing the source hash (L007_ns_above_five_is_a_noop). "
                f"Only ns in 1..{_n - 1} is a real experiment.")
        s = sub(s, "momentum=0.95, ns_steps=5,", f"momentum=0.95, ns_steps={cfg['ns']},")


    # ---- more mechanism families, opening axes the campaign has never touched ----

    # M6 objective/regularisation: z-loss on the logit partition function (PaLM-style).
    if cfg.get("zloss") is not None:
        s = sub(s, """            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                   ignore_index=-1, reduction=reduction)""",
"""            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                   ignore_index=-1, reduction=reduction)
            if self.training and reduction == 'mean':
                loss = loss + ZLOSS_W * (torch.logsumexp(logits.view(-1, logits.size(-1)), dim=-1) ** 2).mean()""")
        s = sub(s, "HEAD_DIM = 128",
                      f"HEAD_DIM = 128\nZLOSS_W = {cfg['zloss']}\n_zl_sum = [0.0, 0.0, 0]")
        s = sub(s, "                loss = loss + ZLOSS_W * (torch.logsumexp(logits.view(-1, logits.size(-1)), dim=-1) ** 2).mean()",
"""                # chunked: a second full-width logsumexp over [B*T, vocab] materialises
                # another logit-sized fp32 tensor. Accumulate in slices instead.
                _flat = logits.view(-1, logits.size(-1))
                _acc = torch.zeros((), device=_flat.device, dtype=torch.float32)
                _CH = 16384
                for _i in range(0, _flat.size(0), _CH):
                    _acc = _acc + (torch.logsumexp(_flat[_i:_i+_CH], dim=-1) ** 2).sum()
                _z = _acc / _flat.size(0)
                _zl_sum[0] += _z.detach(); _zl_sum[1] += loss.detach(); _zl_sum[2] += 1
                loss = loss + ZLOSS_W * _z""")
        obs.append("""_zn = max(_zl_sum[2], 1)
_zm = float(_zl_sum[0]) / _zn
print(f"logz_sq_mean:        {_zm:.6f}")
print(f"zloss_frac_of_total: {ZLOSS_W*_zm/max(float(_zl_sum[1])/_zn,1e-9):.6f}")""")

    # M8 gradient geometry: WHERE the elementwise second moment is applied relative to
    # the polar (Newton-Schulz) iteration. The baseline already runs NorMuon variance
    # reduction, but AFTER orthogonalization, where it can only rescale an update whose
    # direction is already fixed. Three independent sources say the ordering matters and
    # that BEFORE is the better one at short horizons (b0_muon2_second_moment_before_ns,
    # b0_vamuon_variance_modulation_before_ns, b0_adamuon_sign_before_polar_helps_alone);
    # VA-Muon's ordering ablation specifically reports the post-orthogonalization variant
    # sitting BELOW plain Muon early, and this benchmark is entirely early.
    #
    # This is a pure REORDER of blocks that already exist -- no new tensor is allocated
    # and no operation is added, which matters because a registered claim
    # (b1_optimizer_fusion_break_costs_loss) reports a loss regression caused purely by
    # breaking the compiler's fused bf16->fp32->bf16 weight update rather than by any
    # algorithmic change. Keeping the block verbatim keeps that fusion intact.
    if cfg.get("precond") is not None:
        if cfg["precond"] != "pre":
            raise VariantEditError(f"precond must be 'pre' (got {cfg['precond']!r})")
        NORMUON = """    # NorMuon variance reduction
    beta2 = beta2_t.to(g.dtype)
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)
"""
        # remove it from after the polar iteration ...
        s = sub(s, NORMUON, "", "NorMuon block (post-orthogonalization)")
        # ... and reinstate it verbatim on the MOMENTUM, before orthogonalization.
        s = sub(s, """    g = stacked_grads.lerp_(momentum_buffer, momentum)
""", """    g = stacked_grads.lerp_(momentum_buffer, momentum)
""" + NORMUON, "momentum lerp (pre-orthogonalization insertion point)")
        # Activation observable. It must be computed OUTSIDE muon_step_fused: that
        # function is @torch.compile(fullgraph=True) and a print inside it would either
        # graph-break or force a recompile, which is exactly the class of accident that
        # produced a loss regression with no algorithmic cause in the cited claim.
        # second_momentum_buffer now tracks the second moment of the RAW MOMENTUM rather
        # than of the orthogonalized update, and those two live on completely different
        # scales -- an orthogonalized matrix has entries of order 1/sqrt(max(A,B)) by
        # construction, so this statistic moving by orders of magnitude is what proves
        # the reorder actually took effect rather than a flag saying it should have.
        # NOTE: the observable for this mechanism is emitted UNCONDITIONALLY by the
        # instrumentation block below, in the control arm too. Emitting it only in the
        # treatment would leave nothing to compare it against, so "engaged" could not be
        # distinguished from "took some value" -- the mechanism would be unfalsifiable.

    # M7 signal propagation: rotary base (how fast position phases rotate).
    if cfg.get("rope") is not None:
        s = sub(s, "def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):",
                      f"def _precompute_rotary_embeddings(self, seq_len, head_dim, base={cfg['rope']}, device=None):")

    # M8 numerics / signal path: the logit softcap ceiling.
    if cfg.get("softcap") is not None:
        s = sub(s, "        softcap = 15", f"        softcap = {cfg['softcap']}")

    # M9 parameterisation: drop the RMS norm on q/k before attention.
    if cfg.get("noqknorm") is not None:
        s = sub(s, "        q, k = norm(q), norm(k)",
"""        if not hasattr(self, '_qk_probe'): self._qk_probe = []
        if len(self._qk_probe) < 64:
            self._qk_probe.append((q.detach().float().pow(2).mean().sqrt(),
                                   k.detach().float().pow(2).mean().sqrt()))""")
        obs.append("""_m = model._orig_mod if hasattr(model,'_orig_mod') else model
_pr = [p for b in _m.transformer.h for p in getattr(b.attn, '_qk_probe', [])]
if _pr:
    print(f"q_rms_mean:          {sum(float(a) for a,_ in _pr)/len(_pr):.6f}")
    print(f"k_rms_mean:          {sum(float(b) for _,b in _pr)/len(_pr):.6f}")""")

    # M10 stochastic optimisation: global gradient clipping before the optimizer step.
    if cfg.get("clip") is not None:
        # clip_grad_norm_ RETURNS the pre-clip total norm. Without capturing it, a
        # threshold that never binds produces a byte-different variant that is
        # mathematically identical to the control -- a burned slot whose null result is
        # indistinguishable from "the mechanism does nothing". Record the norms so
        # `clip_fired_frac` can say whether the clip engaged at all.
        s = sub(s, "t_start_training = time.time()",
                      "t_start_training = time.time()\n_gn = []")
        s = sub(s, "    optimizer.step()",
                      f"    _gn.append(float(torch.nn.utils.clip_grad_norm_("
                      f"model.parameters(), {cfg['clip']})))\n    optimizer.step()")
        obs.append(f"""if _gn:
    _g = sorted(_gn)
    print(f"grad_norm_p50:       {{_g[len(_g)//2]:.6f}}")
    print(f"grad_norm_p99:       {{_g[int(len(_g)*0.99)]:.6f}}")
    print(f"clip_fired_frac:     {{sum(1 for v in _gn if v > {cfg['clip']})/len(_gn):.6f}}")""")

    # M11 temporal dynamics: constant weight decay instead of the linear decay to zero.
    if cfg.get("wd_const") is not None:
        s = sub(s, """def get_weight_decay(progress):
    return WEIGHT_DECAY * (1 - progress)""",
f"""def get_weight_decay(progress):
    return {cfg['wd_const']}""")

    # M12 temporal dynamics: add an LR warmup the baseline does not have (WARMUP_RATIO=0).
    if cfg.get("warmup") is not None:
        s = sub(s, "WARMUP_RATIO = 0.0", f"WARMUP_RATIO = {cfg['warmup']}")

    # M13 signal propagation: initial value of the embedding-reinjection scalars.
    if cfg.get("x0init") is not None:
        s = sub(s, "        self.x0_lambdas.fill_(0.1)", f"        self.x0_lambdas.fill_({cfg['x0init']})")


    # ---- directions round 1: only what the INDEPENDENT CRITIC cleared ----

    # D2 compile mode. Critic: mediator directly observed, zero VRAM, but step-count CV is
    # 6.96% so it is unresolvable at n=1 -> queued with replicates.
    if cfg.get("compile_mode"):
        s = sub(s, "model = torch.compile(model, dynamic=False)",
                      f"model = torch.compile(model, dynamic=False, mode={cfg['compile_mode']!r})")

    # D4 upper-layer Q/K early-LR suppression. Critic correction: the paper releases at
    # 3-6% of training (forced by 12%), NOT the 30% Stage 1 proposed. Honour the paper.
    if cfg.get("qk_suppress") is not None:
        s = sub(s, """    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm""",
f"""    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        # upper-half Q/K suppressed early, released by QK_RELEASE of the budget
        if group.get("_qk_upper"):
            _rel = min(progress / QK_RELEASE, 1.0) if QK_RELEASE > 0 else 1.0
            group["lr"] = group["lr"] * (QK_ALPHA + (1.0 - QK_ALPHA) * _rel)""")
        s = sub(s, "        optimizer = MuonAdamW(param_groups)",
"""        for _g in param_groups:
            if _g.get('kind') == 'muon':
                _ps = _g['params']
                _g['_qk_upper'] = any(
                    any(p is b.attn.c_q.weight or p is b.attn.c_k.weight for p in _ps)
                    for i, b in enumerate(self.transformer.h) if i >= self.config.n_layer // 2)
        optimizer = MuonAdamW(param_groups)""")
        s = sub(s, "HEAD_DIM = 128",
                      f"HEAD_DIM = 128\nQK_ALPHA = {cfg['qk_suppress']}\nQK_RELEASE = {cfg.get('qk_release', 0.12)}")

    # D9 batch ramp at constant tokens. Critic: score on RAW val_bpb -- it changes steps at
    # constant tokens, so the step-law residual double-counts. LR scaled by sqrt(2).
    if cfg.get("batch_ramp") is not None:
        s = sub(s, "    progress = min(total_training_time / TIME_BUDGET, 1.0)",
"""    progress = min(total_training_time / TIME_BUDGET, 1.0)
    _ramped = progress >= BATCH_RAMP_AT
    grad_accum_steps = _BASE_ACCUM * (2 if _ramped else 1)""")
        s = sub(s, "        group[\"lr\"] = group[\"initial_lr\"] * lrm",
                      "        group[\"lr\"] = group[\"initial_lr\"] * lrm * (2 ** 0.5 if _ramped else 1.0)")
        s = sub(s, "grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd",
                      f"grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd\n_BASE_ACCUM = grad_accum_steps\nBATCH_RAMP_AT = {cfg['batch_ramp']}")

    # --- telemetry: report the MODEL, not the constants (fixes the depth:8 provenance bug) ---
    s = sub(s, 'print(f"depth:            {DEPTH}")',
                  'print(f"depth:            {config.n_layer}")')
    s = sub(s, 'print(f"seed:             {SEED}")', '''print(f"seed:             {SEED}")
_tt = max(total_training_time, 1e-9)
_sd = sorted(_dts)
print(f"n_embd:           {config.n_embd}")
print(f"mlp_ratio:        {cfg_mlp_ratio}")
print(f"n_ve_layers:      {len(model._orig_mod.value_embeds) if hasattr(model,'_orig_mod') else len(model.value_embeds)}")
print(f"tokens_per_step:  {TOTAL_BATCH_SIZE}")
print(f"grad_accum:       {grad_accum_steps}")
print(f"flops_per_token_M:{num_flops_per_token/1e6:.3f}")
print(f"loader_frac:      {_t_loader/_tt:.6f}")
print(f"fwdbwd_frac:      {_t_fwdbwd/_tt:.6f}")
print(f"step_ms_p10:      {_sd[len(_sd)//10]*1000:.2f}")
print(f"step_ms_med:      {_sd[len(_sd)//2]*1000:.2f}")
print(f"vram_reserved_mb: {torch.cuda.max_memory_reserved()/1024/1024:.1f}")
print(f"final_epoch:      {epoch}")''')
    s = sub(s, "HEAD_DIM = 128", f"HEAD_DIM = 128\ncfg_mlp_ratio = {cfg.get('mlp',4)}")
    if obs:
        s = s + "\n# --- activation observables (mechanism-specific) ---\n" + "\n".join(obs) + "\n"
    return s

if __name__ == "__main__":
    cfg = json.loads(sys.argv[1])
    sys.stdout.write(build(cfg))
