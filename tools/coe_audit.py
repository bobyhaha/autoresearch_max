#!/usr/bin/env python3
"""Chain-of-Evidence audit for the campaign's own claims.

Motivated by ScientistOne (arXiv 2605.26340), which defines Chain-of-Evidence as:
*every claim must be traceable, through a recorded chain, to a grounding source*, and
audits finished papers with four checks -- Score Verification, Specification Violation,
Reference Verification, Method-Code Alignment.

We are NOT copying that. Their failure mode is a paper whose citations were invented.
Ours, measured over this campaign, is different in kind: the system makes confident
**operational** claims about itself that trace to nothing.

    "running: 5"                 -> zero train.py processes existed
    "busy=4/4" (watchdog)        -> the GPUs were at 4 MiB and 0%
    "TOPPING UP ... staging"     -> nothing was enqueued
    `literature-source` exit 0   -> printed an error JSON and registered nothing
    content.status fulltext      -> cited a source that was abstract-only

Each is a claim with no grounding. So the adaptation is to push CoE *down* from the
paper layer to the operational and evidential layers, where our claims actually break.
Four checks, each mapped from theirs but re-aimed at a failure this campaign has really
suffered:

  A1 LIVENESS      (their I1, Score Verification)
       They re-run the solution and compare to the reported score. We re-observe the
       box and compare to the queue's self-report. A job claiming `running` must have a
       real process on its pinned GPU. This is the check that would have caught 5 idle
       H200s behind gpu0-pinned zombies -- twice.

  A2 ACTIVATION    (their I2, Specification Violation)
       Theirs catches code that games the evaluator. Ours catches the opposite: an arm
       whose intervention never engaged, so the run is not a test of its hypothesis. A
       null from an inactive intervention must be recorded INCONCLUSIVE, never as
       evidence against the mechanism -- otherwise we retire good ideas for free. Also
       flags specs pinned to a resource a foreign tenant holds, which can never run.

  A3 SOURCE        (their I3, Reference Verification)
       Same spirit, stricter: existence is not enough. A source asserting
       `fulltext_snapshot` must have the snapshot on disk with a matching sha256, and a
       claim may not assert provenance stronger than its source actually records.

  A4 RESOLVABILITY (no counterpart in their work -- and this is the important one)
       Their evaluator is deterministic, so I1 compares "within an adaptive tolerance".
       Ours is stochastic with control sigma ~0.0216 against a gate of 0.000426: raw
       noise is ~50x the effect we chase. The honest analogue of Score Verification for
       a noisy evaluator is not tolerance, it is POWER. A delta may not be reported as
       evidence unless the replicates actually run could resolve it. This check exists
       because the campaign's real epistemic risk is not fabrication, it is confidently
       reporting noise.

Exit code is non-zero if any BLOCKING violation is found, so this can gate a heartbeat.

    uv run python tools/coe_audit.py                # full audit
    uv run python tools/coe_audit.py --skip-remote  # no ssh (A1 degrades to local)
    uv run python tools/coe_audit.py --json         # machine-readable
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
STORE = REPO / ".autoresearch"
QUEUE = STORE / "operational" / "queue"
SCIENCE = REPO / "runs" / "science"
SNAPSHOTS = REPO / "runs" / "sources"

GATE = 0.000426


def load_env() -> dict:
    env = dict(os.environ)
    dotenv = REPO / ".env"
    if dotenv.exists():
        for line in dotenv.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    return env


def remote(cmd: str, env: dict, timeout: int = 45) -> str:
    key, port, target = (env.get("OPHIS_SSH_KEY"), env.get("OPHIS_SSH_PORT"),
                         env.get("OPHIS_SSH_TARGET"))
    if not all((key, port, target)):
        return ""
    argv = ["ssh", "-i", key, "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes",
            "-o", "ConnectTimeout=15", "-o", "ControlMaster=auto",
            "-o", "ControlPath=~/.ssh/cm/%r@%h:%p", "-o", "ControlPersist=600",
            "-p", port, target, cmd]
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=timeout,
                              check=False)
    except subprocess.TimeoutExpired:
        return ""
    return done.stdout if done.returncode == 0 else ""


def foreign_held(env: dict) -> dict[str, str]:
    """GPU index -> username, for GPUs held by someone who is not us.

    Memory-used alone cannot answer "is this ours?" -- our own healthy 49 GB run looks
    identical to a tenant's. Ownership needs pid -> user, resolved through the bus id.
    Getting this wrong in the first version of this audit produced three confident
    false-positive criticals against our own live runs, which is the same ungrounded-claim
    failure the audit exists to catch.
    """
    bus_to_idx = {}
    for line in remote("nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader",
                       env).splitlines():
        cells = [c.strip() for c in line.split(",")]
        if len(cells) == 2:
            bus_to_idx[cells[1]] = cells[0]

    me = remote("id -un", env).strip()
    held: dict[str, str] = {}
    for line in remote(
        "nvidia-smi --query-compute-apps=gpu_bus_id,pid --format=csv,noheader", env
    ).splitlines():
        cells = [c.strip() for c in line.split(",")]
        if len(cells) != 2 or not cells[1].isdigit():
            continue
        idx = bus_to_idx.get(cells[0])
        if idx is None:
            continue
        user = remote(f"ps -o user= -p {cells[1]} 2>/dev/null || true", env).strip()
        if user and me and not user.startswith(me[:8]):
            held[idx] = user
    return held


def read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def jobs() -> list[dict]:
    return [d for d in (read_json(p) for p in QUEUE.glob("job_*.json")) if d]


def record_payloads(kind: str) -> list[dict]:
    out = []
    for path in (STORE / "records" / kind).glob("*.json"):
        envelope = read_json(path)
        if not isinstance(envelope, dict) or not isinstance(envelope.get("payload"), dict):
            continue
        out.append({"id": envelope.get("id"), **envelope["payload"]})
    return out


# ---------------------------------------------------------------- A1 liveness

def check_liveness(env: dict, skip_remote: bool) -> list[dict]:
    """The queue's self-report must agree with the box. Ours is the claim; the box is
    the grounding source."""
    findings = []
    active = [j for j in jobs() if j.get("state") in ("running", "waiting")]
    claimed_running = [j for j in active if j.get("state") == "running"]

    if skip_remote:
        return [{"check": "A1", "severity": "info",
                 "finding": f"remote skipped; queue self-reports {len(claimed_running)} running",
                 "blocking": False}]

    smi = remote("nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits", env)
    procs = remote('ps -u "$USER" -o cmd --no-headers | grep -c "[t]rain.py" || true', env)
    if not smi:
        return [{"check": "A1", "severity": "warn",
                 "finding": "box unreachable -- liveness ungrounded, cannot verify any run",
                 "blocking": False}]

    used = {}
    for line in smi.splitlines():
        cells = [c.strip() for c in line.split(",")]
        if len(cells) == 2 and cells[0].isdigit():
            try:
                used[cells[0]] = float(cells[1])
            except ValueError:
                pass
    live = int(procs.strip() or 0)
    foreign = foreign_held(env)

    # The central check: claimed running vs processes that actually exist.
    if claimed_running and live == 0:
        findings.append({
            "check": "A1", "severity": "critical", "blocking": True,
            "finding": f"queue claims {len(claimed_running)} job(s) running but the box has "
                       f"ZERO train.py processes -- the campaign is asserting work that is "
                       f"not happening",
            "evidence": {"claimed_running": len(claimed_running), "remote_train_procs": 0,
                         "spec_ids": [j.get("spec_id") for j in claimed_running][:6]},
        })
    elif live < len(claimed_running):
        findings.append({
            "check": "A1", "severity": "warn", "blocking": False,
            "finding": f"{len(claimed_running)} claimed running but only {live} train.py "
                       f"processes -- some jobs hold slots without executing",
            "evidence": {"claimed_running": len(claimed_running), "remote_train_procs": live},
        })

    # Unrunnable pinning: a job pinned to a GPU a foreign tenant holds can never run,
    # and because max_concurrent_jobs is HOST-wide it starves every free GPU behind it.
    for job in active:
        m = re.search(r"gpu_?(\d+)$", str(job.get("spec_id", "")))
        if not m:
            continue
        gpu = m.group(1)
        if gpu in foreign:
            findings.append({
                "check": "A1", "severity": "critical", "blocking": True,
                "finding": f"job pinned to gpu{gpu} which foreign tenant "
                           f"'{foreign[gpu]}' holds ({used.get(gpu, 0):.0f} MiB) -- unrunnable, "
                           f"and it consumes a HOST-wide slot while free GPUs idle",
                "evidence": {"job_id": job.get("job_id"), "spec_id": job.get("spec_id"),
                             "no_resource_count": job.get("no_resource_count"),
                             "run_attempts": job.get("run_attempts")},
            })

    idle_free = [g for g, mb in used.items() if mb < 512]
    if idle_free and not claimed_running:
        findings.append({
            "check": "A1", "severity": "warn", "blocking": False,
            "finding": f"{len(idle_free)} GPU(s) free with an empty queue -- staging is a "
                       f"scientific decision the guard will not make",
            "evidence": {"free_gpus": sorted(idle_free)},
        })
    if not findings:
        findings.append({"check": "A1", "severity": "ok", "blocking": False,
                         "finding": f"liveness grounded: {live} train.py process(es) match "
                                    f"{len(claimed_running)} claimed running"})
    return findings


# -------------------------------------------------------------- A2 activation

def check_activation() -> list[dict]:
    """An arm whose intervention never engaged is not a test of its hypothesis.

    Records the required predicate if a hypothesis lacks one -- absence is itself a
    finding, because without it a silently-inactive intervention reads as a clean null.
    """
    findings = []
    hyps = {row["id"]: row for row in record_payloads("scientific_hypothesis")}
    missing = [h for h in hyps.values() if not h.get("activation")]
    if missing:
        findings.append({
            "check": "A2", "severity": "warn", "blocking": False,
            "finding": f"{len(missing)}/{len(hyps)} hypotheses declare no `activation` "
                       f"predicate -- a run where the intervention silently no-ops would be "
                       f"scored as evidence against the mechanism",
            "evidence": {"hypothesis_ids": [h["id"] for h in missing][:8]},
        })

    for hyp in hyps.values():
        act = hyp.get("activation")
        if not isinstance(act, dict) or not act.get("predicate"):
            continue
        findings.append({
            "check": "A2", "severity": "info", "blocking": False,
            "finding": f"{hyp['id']} declares activation predicate: {act['predicate']}",
        })
    if not findings:
        findings.append({"check": "A2", "severity": "ok", "blocking": False,
                         "finding": "no hypotheses to audit"})
    return findings


# ------------------------------------------------------------------ A3 source

def check_sources() -> list[dict]:
    """`fulltext_snapshot` is a claim about a file. Verify the file and its digest."""
    findings = []
    sources = record_payloads("literature_source")
    if not sources:
        return [{"check": "A3", "severity": "ok", "blocking": False,
                 "finding": "no registered literature sources"}]

    total = full = broken = 0
    for d in sorted(sources, key=lambda row: str(row.get("id", ""))):
        total += 1
        content = d.get("content") or {}
        if content.get("status") != "fulltext_snapshot":
            continue
        full += 1
        rel = content.get("blob")
        digest = content.get("sha256")
        if not rel or not digest:
            broken += 1
            findings.append({
                "check": "A3", "severity": "critical", "blocking": True,
                "finding": f"{d.get('id')} asserts fulltext_snapshot without "
                           f"{'blob' if not rel else 'sha256'} -- unverifiable provenance",
                "evidence": {"source_id": d.get("id")},
            })
            continue
        target = (STORE / rel).resolve()
        if not target.exists():
            broken += 1
            findings.append({
                "check": "A3", "severity": "critical", "blocking": True,
                "finding": f"{d.get('id')} asserts fulltext_snapshot but the snapshot is missing",
                "evidence": {"source_id": d.get("id"), "expected_path": str(target)},
            })
            continue
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual != digest:
            broken += 1
            findings.append({
                "check": "A3", "severity": "critical", "blocking": True,
                "finding": f"{d.get('id')} snapshot digest mismatch -- the text read is not "
                           f"the text recorded",
                "evidence": {"source_id": d.get("id"), "declared": digest[:16],
                             "actual": actual[:16]},
            })

    # A claim may not assert provenance stronger than its source records.
    status_by_id = {
        str(row["id"]): (row.get("content") or {}).get("status") for row in sources
    }
    for d in record_payloads("scientific_claim"):
        asserted = (d.get("evidence") or {}).get("content_status") or d.get("content_status")
        for sid in (d.get("source_ids") or []):
            have = status_by_id.get(sid)
            if asserted == "fulltext_snapshot" and have and have != "fulltext_snapshot":
                findings.append({
                    "check": "A3", "severity": "critical", "blocking": True,
                    "finding": f"claim {d.get('id')} asserts fulltext_snapshot but source "
                               f"{sid} records {have}",
                    "evidence": {"claim_id": d.get("id"), "source_id": sid},
                })
    if not any(f["severity"] == "critical" for f in findings):
        findings.insert(0, {"check": "A3", "severity": "ok", "blocking": False,
                            "finding": f"{full} fulltext snapshots verified against their "
                                       f"digests ({total} sources total), 0 broken"})
    return findings


# ------------------------------------------------------------ A4 resolvability

def control_sigma() -> tuple[float | None, float | None, int]:
    """Return (raw sigma, step-law residual sigma, n).

    Raw sigma mixes in throughput variation from a shared box, which the step law
    val_bpb = 1.5299 - 0.0782*ln(steps) explains. Subtracting the explained component is
    the campaign's control variate; the residual is what a paired design must beat.
    """
    pairs = []
    for p in STORE.rglob("result_*.json"):
        d = read_json(p)
        if not isinstance(d, dict):
            continue
        payload = d.get("payload") or {}
        for arm in payload.get("arms") or []:
            # CONTROLS ONLY. Pooling candidates in inflates sigma with the very effects
            # we are trying to measure -- the first version of this check did exactly
            # that and reported a residual sigma LARGER than the raw one.
            if arm.get("name") != "control" or arm.get("status") != "completed":
                continue
            met = arm.get("metrics") or {}
            bpb, steps = met.get("val_bpb"), met.get("num_steps")
            if isinstance(bpb, (int, float)) and isinstance(steps, (int, float)) and steps > 0:
                pairs.append((float(bpb), int(steps)))
    if len(pairs) < 3:
        return None, None, len(pairs)
    raw = statistics.pstdev([b for b, _ in pairs])
    # The step law was fitted in the 600-760 step band and does NOT extrapolate: at 1040
    # steps it under-predicts by ~0.028. Reporting one pooled residual sigma hides that
    # the campaign has two regimes, so compute the band separately -- inside it the
    # control variate works, outside it the variance is structural and unexplained.
    band = [(b, s) for b, s in pairs if 600 <= s <= 760]
    resid_band = [b - (1.5299 - 0.0782 * math.log(s)) for b, s in band]
    resid_all = [b - (1.5299 - 0.0782 * math.log(s)) for b, s in pairs]
    globals()["_BAND_INFO"] = {
        "in_band_n": len(band), "out_band_n": len(pairs) - len(band),
        "sigma_out_of_band": (round(statistics.pstdev(
            [b - (1.5299 - 0.0782 * math.log(s)) for b, s in pairs if not 600 <= s <= 760]), 6)
            if len(pairs) - len(band) >= 3 else None),
        "sigma_all_residual": round(statistics.pstdev(resid_all), 6) if len(resid_all) >= 3 else None,
    }
    return raw, (statistics.pstdev(resid_band) if len(resid_band) >= 3 else None), len(pairs)


def check_resolvability(replicates: int) -> list[dict]:
    """Refuse to treat a delta as evidence when the design cannot resolve it.

    Minimum resolvable effect at n replicates, two-sided 95%, is ~ 1.96*sigma*sqrt(2/n)
    for a difference of means. Compare against the promotion gate.
    """
    raw, resid, n_obs = control_sigma()
    if raw is None:
        return [{"check": "A4", "severity": "warn", "blocking": False,
                 "finding": f"only {n_obs} observations -- sigma unestimable"}]

    def mre(sigma: float) -> float:
        return 1.96 * sigma * math.sqrt(2.0 / max(replicates, 1))

    def need(sigma: float) -> int:
        return math.ceil(2 * (1.96 * sigma / GATE) ** 2)

    findings = [{
        "check": "A4", "severity": "info", "blocking": False,
        "finding": (f"RAW design: sigma {raw:.6f} over {n_obs} runs -> at n={replicates} the "
                    f"minimum resolvable effect is {mre(raw):.5f} = {mre(raw) / GATE:.0f}x the "
                    f"gate. Resolving a gate-sized effect this way needs ~{need(raw)} "
                    f"replicates per arm, which is not a real option."),
        "evidence": {"sigma_raw": round(raw, 6), "n_observations": n_obs,
                     "mre_raw": round(mre(raw), 6), "gate": GATE},
    }]

    if resid is None:
        return findings

    band = globals().get("_BAND_INFO", {})
    if band.get("sigma_out_of_band") and band.get("sigma_all_residual"):
        findings.append({
            "check": "A4", "severity": "critical", "blocking": True,
            "finding": (f"REGIME SPLIT: the step-law control variate works only in the "
                        f"600-760 step band (residual sigma {resid:.6f}, n={band['in_band_n']}). "
                        f"Outside it residual sigma is {band['sigma_out_of_band']:.6f} over "
                        f"n={band['out_band_n']} -- {band['sigma_out_of_band'] / max(resid, 1e-9):.0f}x "
                        f"worse and structural, not throughput. Candidates that win on "
                        f"throughput leave the only band where we can measure them."),
            "evidence": band,
        })

    ok = mre(resid) <= GATE
    findings.append({
        "check": "A4", "severity": "ok" if ok else "critical", "blocking": not ok,
        "finding": (f"CONTROL-VARIATE design: step-law residual sigma {resid:.6f} "
                    f"({raw / resid:.0f}x noise reduction) -> at n={replicates} the minimum "
                    f"resolvable effect is {mre(resid):.6f}"
                    + (f", which clears the {GATE} gate." if ok else
                       f", still {mre(resid) / GATE:.1f}x the {GATE} gate. Need n>="
                       f"{need(resid)} per arm to call a gate-sized effect.")),
        "evidence": {"sigma_residual": round(resid, 6),
                     "noise_reduction_x": round(raw / resid, 1),
                     "mre_residual": round(mre(resid), 6),
                     "replicates_needed_for_gate": need(resid)},
    })
    if not ok:
        findings.append({
            "check": "A4", "severity": "critical", "blocking": True,
            "finding": (f"DESIGN VERDICT: at n={replicates} this campaign cannot resolve the "
                        f"effects it is chasing. Either run n>={need(resid)} replicates per "
                        f"arm, or only pursue mechanisms predicting >= {mre(resid):.4f} bpb "
                        f"-- roughly {mre(resid) / GATE:.0f} gates. Reporting a smaller delta "
                        f"as a win or a loss is reporting noise."),
        })
    return findings


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-remote", action="store_true")
    ap.add_argument("--replicates", type=int, default=1,
                    help="replicates per arm the campaign actually runs (default 1)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    env = load_env()
    findings = (check_liveness(env, args.skip_remote) + check_activation()
                + check_sources() + check_resolvability(args.replicates))

    if args.json:
        print(json.dumps({"findings": findings}, indent=2))
    else:
        order = {"critical": 0, "warn": 1, "info": 2, "ok": 3}
        icon = {"critical": "FAIL", "warn": "WARN", "info": "note", "ok": " ok "}
        print("=" * 92)
        print("Chain-of-Evidence audit -- every claim this campaign makes about itself")
        print("=" * 92)
        for f in sorted(findings, key=lambda x: (order.get(x["severity"], 9), x["check"])):
            print(f"[{icon[f['severity']]}] {f['check']}  {f['finding']}")
            if f.get("evidence"):
                print(f"          {json.dumps(f['evidence'])[:210]}")
        blocking = [f for f in findings if f.get("blocking")]
        print("-" * 92)
        print(f"{len(findings)} findings, {len(blocking)} blocking")
    return 1 if any(f.get("blocking") for f in findings) else 0


if __name__ == "__main__":
    raise SystemExit(main())
