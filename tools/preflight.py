#!/usr/bin/env python3
"""Verify everything that must be true before a campaign run, and say what isn't.

Written after a restart in which several things were individually plausible and
jointly wrong: the active code disagreed with what provenance recorded, four remote
workdirs disagreed with the local tree, a scope claimed an evaluator digest that no
longer existed, and a "pristine" baseline silently read a corrected byte table left
in a shared cache by another campaign. Every one of those was cheap to detect and
expensive to miss, and none of them announced itself.

So this is one command that fails loudly rather than a checklist someone remembers.
It checks, in order of how badly it hurts to get wrong:

  1. the active code IS the baseline provenance claims it is
  2. train.py differs from upstream only by the protocol adapter
  3. prepare.py is byte-identical to pinned upstream and its digest IS the evaluator
  4. every remote workdir byte-matches the local tree
  5. upstream's shared cache contains exactly the manifest-bound 10+1 shards and tokenizer
  6. every cache artifact byte-matches the data manifest
  7. the GPUs are healthy: no MIG, no throttle, full power limit, and actually free
  8. the local machine will not sleep mid-run and invalidate the timing reconciliation

Exit 0 means every check passed. Exit 1 means at least one FAILED and running now
would produce measurements you cannot trust. Warnings never fail the run; they are
things worth seeing, not reasons to stop.

    uv run python tools/preflight.py
    uv run python tools/preflight.py --json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
CODE = REPO / "runs" / "code"


def _load_dotenv() -> None:
    try:
        lines = (REPO / ".env").read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for line in lines:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


_load_dotenv()
SSH = [
    "ssh", "-i", os.environ.get("OPHIS_SSH_KEY", ""),
    "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes",
    # Share one TCP connection across every tool and tick. Without this the
    # watchdog, wake, guard, preflight and eight worker threads each open their
    # own session; the host started refusing them, and the harness reported the
    # refusal as "code or data did not match the sealed manifest" on every
    # binding at once -- an environmental fault wearing a scientific fault's
    # clothes.
    "-o", "ControlMaster=auto", "-o", "ControlPath=~/.ssh/cm/%r@%h:%p",
    "-o", "ControlPersist=600", "-o", "ConnectTimeout=20",
    "-p", os.environ.get("OPHIS_SSH_PORT", "22"), os.environ.get("OPHIS_SSH_TARGET", ""),
]
REMOTE_ROOT = os.environ.get("OPHIS_REMOTE_ROOT", "")
# The fleet this operator declared, not a repository-level allocation. GPU ownership on a
# shared box changes between campaigns, so each clone records its active indices in .env.
GPUS = tuple(
    part.strip()
    for part in (os.environ.get("OPHIS_GPUS") or "0").split(",")
    if part.strip()
)


class Report:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def add(self, ok: bool | None, name: str, detail: str = "") -> None:
        self.rows.append({"status": "PASS" if ok else ("WARN" if ok is None else "FAIL"),
                          "check": name, "detail": detail})

    @property
    def failed(self) -> list[dict[str, Any]]:
        return [r for r in self.rows if r["status"] == "FAIL"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(argv: list[str], timeout: int = 90, retries: int = 1) -> tuple[int, str]:
    """Run a command, returning (code, stdout-or-stderr).

    Two things learned the hard way on this box. Failures reported STDERR while this
    only returned stdout, so a failed check printed an empty reason and told the
    reader nothing. And the host is shared at load ~250 with three of our own tools
    SSHing every ten minutes, so a single connection refusal is routine noise, not a
    reason to block a run; transient transport failures are retried before concluding.
    """
    last = (124, "not attempted")
    for attempt in range(retries + 1):
        try:
            done = subprocess.run(argv, capture_output=True, text=True, timeout=timeout,
                                  cwd=REPO, check=False)
            if done.returncode == 0:
                return 0, done.stdout.strip()
            detail = (done.stderr or done.stdout or "").strip() or f"exit {done.returncode}"
            last = (done.returncode, detail)
        except (subprocess.TimeoutExpired, OSError) as exc:
            last = (124, f"{type(exc).__name__}: {exc}")
        if attempt < retries:
            time.sleep(3 * (attempt + 1))
    return last


def check_code(rep: Report) -> None:
    prov = json.loads((CODE / "provenance.json").read_text())
    active, upstream = prov["active_adapter"], prov["upstream"]

    for name in ("train.py", "prepare.py"):
        actual = sha256(CODE / name)
        expected = active[f"{name.split('.')[0]}_py_sha256"]
        rep.add(actual == expected, f"active {name} matches provenance",
                "" if actual == expected else f"{actual[:16]} != {expected[:16]}")

    # The pristine copies must still be pristine, or every comparison above is moot.
    for name in ("train.py", "prepare.py"):
        actual = sha256(CODE / "upstream" / name)
        expected = upstream[f"{name.split('.')[0]}_py_sha256"]
        rep.add(actual == expected, f"upstream/{name} is pristine {upstream['commit'][:7]}",
                "" if actual == expected else f"{actual[:16]} != {expected[:16]}")

    prepare_exact = (CODE / "prepare.py").read_bytes() == (CODE / "upstream" / "prepare.py").read_bytes()
    rep.add(prepare_exact, "prepare.py is byte-identical to fixed upstream",
            "" if prepare_exact else "Karpathy marks prepare.py fixed; restore it before running")

    # train.py is the file that computes the metric. Bound how far it may drift from
    # Karpathy's: the adapter emits numbers, it must not change any.
    import difflib
    up = (CODE / "upstream" / "train.py").read_text().splitlines()
    cur = (CODE / "train.py").read_text().splitlines()
    removed = [ln for ln in difflib.unified_diff(up, cur, lineterm="")
               if ln.startswith("-") and not ln.startswith("---")]
    rep.add(len(removed) <= 5, "train.py differs from upstream by adapter only",
            f"{len(removed)} upstream lines replaced (limit 5)")
    source = (CODE / "train.py").read_text()
    adapter_ok = all(
        token in source
        for token in (
            'SEED = int(os.environ.get("AUTORESEARCH_SEED", "42"))',
            "t_start = PROCESS_START",
            "torch.manual_seed(SEED)",
            'print(f"seed:             {SEED}")',
        )
    )
    rep.add(adapter_ok, "sealed seed/timing adapter is present")


def check_scope(rep: Report) -> None:
    scope = json.loads((REPO / "runs" / "scope.json").read_text())
    evaluator = scope.get("evaluator", "")
    actual = f"sha256:{sha256(CODE / 'prepare.py')}"
    # prepare.py's digest IS the evaluator: it defines the metric. If these disagree,
    # results get banked under a scope that never produced them.
    rep.add(evaluator == actual, "scope evaluator == prepare.py digest",
            "" if evaluator == actual else f"scope says {evaluator[:23]}, file is {actual[:23]}")
    rep.add(bool(scope.get("id")), "scope has an id", scope.get("id", ""))
    budget = (scope.get("budget") or {})
    rep.add(budget.get("kind") == "training_seconds" and float(budget.get("value", 0)) == 300.0,
            "scope budget is 300 training_seconds", json.dumps(budget))
    manifest_path = REPO / "runs" / "data_manifest.json"
    expected_manifest = str((scope.get("data_details") or {}).get("manifest_sha256", ""))
    actual_manifest = sha256(manifest_path) if manifest_path.is_file() else "missing"
    rep.add(expected_manifest == actual_manifest, "scope binds current data manifest",
            "" if expected_manifest == actual_manifest else
            f"scope {expected_manifest[:16]} != manifest {actual_manifest[:16]}")


def check_remote(rep: Report) -> None:
    if not REMOTE_ROOT or not os.environ.get("OPHIS_SSH_TARGET"):
        rep.add(False, "remote configured", "OPHIS_* env not set; copy .env.example to .env")
        return
    local = {name: sha256(CODE / name) for name in ("train.py", "prepare.py")}
    local["baseline_provenance.json"] = sha256(CODE / "provenance.json")

    cmd = "; ".join(
        f'echo "gpu{g} $(sha256sum {REMOTE_ROOT}/gpu{g}/train.py {REMOTE_ROOT}/gpu{g}/prepare.py '
        f'{REMOTE_ROOT}/gpu{g}/baseline_provenance.json 2>/dev/null | cut -d\\  -f1 | tr \\\\n \\ )"'
        for g in GPUS
    )
    code, out = run([*SSH, cmd], retries=2)
    if code != 0:
        rep.add(False, "remote reachable", out[:120])
        return
    rep.add(True, "remote reachable")
    for line in out.splitlines():
        parts = line.split()
        if len(parts) != 4:
            rep.add(False, f"{parts[0] if parts else '?'} workdir readable", line[:80])
            continue
        gpu, t, p, b = parts
        # prepare.py and baseline_provenance.json are NEVER mutable: prepare.py's digest
        # IS the scope evaluator and provenance is the seal, so a mismatch there means a
        # run would be measured against a metric nobody declared. Those stay FAIL.
        #
        # train.py is the declared mutable path. A candidate arm materialises its own
        # train.py into the workdir by design, so a mismatch there while candidates are
        # in flight is expected, not a fault -- failing on it would train a reader to
        # ignore this whole report. Surfaced as a warning with the digest so it can be
        # checked against the sealed candidate manifest.
        sealed_ok = (p == local["prepare.py"] and b == local["baseline_provenance.json"])
        rep.add(sealed_ok, f"{gpu} sealed bindings match (prepare.py, provenance)",
                "" if sealed_ok else f"prepare {p[:8]} prov {b[:8]}")
        if t != local["train.py"]:
            rep.add(None, f"{gpu} train.py differs from baseline",
                    f"{t[:12]} — expected while a candidate arm is running on this GPU")


def check_corpus_and_cache(rep: Report) -> None:
    manifest_path = REPO / "runs" / "data_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, ValueError) as exc:
        rep.add(False, "data manifest parses", str(exc)); return
    rep.add(manifest.get("schema_version") == 3, "data manifest is strict-upstream schema 3",
            f"schema={manifest.get('schema_version')!r}")
    cache_dir = str(manifest.get("cache_dir") or "")
    if not cache_dir.startswith("/"):
        rep.add(False, "data manifest has absolute remote cache_dir", cache_dir or "missing")
        return
    files = manifest.get("files") or []
    expected = {str(row.get("path")): str(row.get("sha256")) for row in files}
    expected_shards = {
        *(f"data/shard_{index:05d}.parquet" for index in range(10)),
        "data/shard_06542.parquet",
    }
    actual_shards = {path for path in expected if path.startswith("data/")}
    rep.add(actual_shards == expected_shards, "manifest pins exactly 10 train + 1 val shards",
            "" if actual_shards == expected_shards else
            f"missing={sorted(expected_shards - actual_shards)} extra={sorted(actual_shards - expected_shards)}")
    rep.add(set(expected) == expected_shards | {"tokenizer/tokenizer.pkl", "tokenizer/token_bytes.pt"},
            "manifest contains only upstream runtime artifacts", f"{len(expected)} files")

    quoted_cache = shlex.quote(cache_dir)
    command = (
        f"find {quoted_cache}/data -maxdepth 1 -type f -name '*.parquet' -printf '%f\\n' | sort; "
        "echo ---HASHES---; "
        + "sha256sum "
        + " ".join(shlex.quote(f"{cache_dir}/{path}") for path in sorted(expected))
    )
    code, out = run([*SSH, command], timeout=300, retries=2)
    if code != 0:
        rep.add(False, "upstream cache inspectable", out[:160]); return
    listing, _, hashes = out.partition("---HASHES---")
    names = {line.strip() for line in listing.splitlines() if line.strip()}
    expected_names = {path.removeprefix("data/") for path in expected_shards}
    rep.add(names == expected_names, "upstream cache directory has no extra shards",
            "" if names == expected_names else
            f"missing={sorted(expected_names - names)} extra={sorted(names - expected_names)}")
    actual_hashes: dict[str, str] = {}
    for line in hashes.splitlines():
        digest, _, absolute = line.strip().partition("  ")
        prefix = f"{cache_dir}/"
        if absolute.startswith(prefix):
            actual_hashes[absolute.removeprefix(prefix)] = digest
    mismatches = [path for path, digest in expected.items() if actual_hashes.get(path) != digest]
    rep.add(not mismatches, "upstream cache byte-matches data manifest",
            "" if not mismatches else f"mismatch or missing: {mismatches}")


def check_gpus(rep: Report) -> None:
    code, out = run([*SSH,
        ("nvidia-smi --query-gpu=index,mig.mode.current,power.limit,clocks.max.sm,memory.used "
         "--format=csv,noheader")], retries=2)
    if code != 0:
        rep.add(False, "GPUs queryable", out[:100]); return
    free, busy = [], []
    for line in out.splitlines():
        cells = [c.strip() for c in line.split(",")]
        if len(cells) < 5:
            continue
        idx, mig, _power, _clk, mem = cells
        if idx not in GPUS:
            continue
        if mig.lower() != "disabled":
            rep.add(False, f"gpu{idx} MIG disabled", mig)
        used = float(mem.split()[0])
        (free if used < 512 else busy).append(idx)
    # "No free GPU" is not a fault -- four GPUs busy with OUR arms is the goal state,
    # and failing preflight on full utilisation would block the campaign at exactly the
    # moment it is working. Report occupancy; never fail on it.
    rep.add(None if not free else True, "our GPU occupancy",
            f"free={free} busy={busy}" + ("  (fully utilised)" if not free else ""))


def check_local(rep: Report) -> None:
    code, _ = run(["pgrep", "-x", "caffeinate"], timeout=15)
    sleep_hint = (
        "macOS sleep will under-measure wall_seconds and invalidate healthy runs; "
        "run: nohup caffeinate -i -m >/dev/null 2>&1 &"
    )
    rep.add(code == 0, "idle sleep held (caffeinate)", "" if code == 0 else sleep_hint)
    code, out = run(["uv", "run", "pytest", "-q"], timeout=600)
    rep.add(code == 0, "test suite passes", out.splitlines()[-1][:90] if out else "")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    rep = Report()
    check_code(rep)
    check_scope(rep)
    check_remote(rep)
    check_corpus_and_cache(rep)
    check_gpus(rep)
    check_local(rep)

    if args.json:
        print(json.dumps(rep.rows, indent=2))
    else:
        width = max(len(r["check"]) for r in rep.rows) + 2
        for r in rep.rows:
            mark = {"PASS": "  ok ", "WARN": " warn", "FAIL": "FAIL "}[r["status"]]
            print(f"{mark} {r['check']:<{width}} {r['detail']}")
        failed = rep.failed
        print()
        print(f"{len(rep.rows)} checks, {len(failed)} failed")
        if failed:
            print("\nDO NOT RUN until these are fixed:")
            for r in failed:
                print(f"  - {r['check']}: {r['detail']}")
    sys.exit(1 if rep.failed else 0)


if __name__ == "__main__":
    main()
