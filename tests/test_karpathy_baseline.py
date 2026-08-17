from __future__ import annotations

import ast
import difflib
import hashlib
import json
from pathlib import Path

from autoresearch.protocol import normalize_scope

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "runs" / "code"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_pristine_karpathy_source_and_runtime_lock_are_pinned() -> None:
    provenance = json.loads((CODE / "provenance.json").read_text())
    upstream = provenance["upstream"]
    assert upstream["repository"] == "https://github.com/karpathy/autoresearch.git"
    assert upstream["commit"] == "228791fb499afffb54b46200aca536f79142f117"
    assert _sha256(CODE / "upstream" / "train.py") == upstream["train_py_sha256"]
    assert _sha256(CODE / "upstream" / "prepare.py") == upstream["prepare_py_sha256"]
    assert _sha256(CODE / "pyproject.toml") == upstream["pyproject_toml_sha256"]
    assert _sha256(CODE / "uv.lock") == upstream["uv_lock_sha256"]


def test_active_baseline_is_auditable_and_satisfies_the_v2_contract() -> None:
    provenance = json.loads((CODE / "provenance.json").read_text())
    active = provenance["active_adapter"]
    assert _sha256(CODE / "train.py") == active["train_py_sha256"]
    assert _sha256(CODE / "prepare.py") == active["prepare_py_sha256"]

    train_source = (CODE / "train.py").read_text()
    prepare_source = (CODE / "prepare.py").read_text()
    ast.parse(train_source)
    ast.parse(prepare_source)
    assert 'SEED = int(os.environ.get("AUTORESEARCH_SEED", "42"))' in train_source
    assert "torch.manual_seed(SEED)" in train_source
    assert "torch.cuda.manual_seed(SEED)" in train_source
    assert 'print(f"seed:             {SEED}")' in train_source

    # Karpathy marks prepare.py fixed. It must be byte-for-byte identical to the
    # pinned upstream source; cache isolation and corpus validation belong in the
    # operational preflight, never in this scientific baseline.
    assert (CODE / "prepare.py").read_bytes() == (CODE / "upstream" / "prepare.py").read_bytes()
    assert 'CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")' in prepare_source
    # The evaluator digest in the scope template IS prepare.py's digest, so a change
    # here without a scope bump would silently redefine the metric.
    assert _sha256(CODE / "prepare.py") == active["prepare_py_sha256"]

    # train.py may differ from upstream ONLY by the protocol adapter. Guard the
    # size of that difference so an unrelated change cannot ride along unnoticed.
    upstream_train = (CODE / "upstream" / "train.py").read_text().splitlines()
    added = [
        line
        for line in difflib.unified_diff(upstream_train, train_source.splitlines(), lineterm="")
        if line.startswith("-") and not line.startswith("---")
    ]
    assert len(added) <= 5, f"adapter removes more upstream lines than expected: {added}"


def test_karpathy_scope_and_execution_templates_bind_the_baseline() -> None:
    scope = json.loads((ROOT / "templates" / "scope_karpathy_autoresearch.json").read_text())
    normalized = normalize_scope(scope)
    assert normalized["budget"] == {"kind": "training_seconds", "value": 300.0}
    assert normalized["metric"] == {"name": "val_bpb", "direction": "minimize"}
    assert normalized["evaluator"] == f"sha256:{_sha256(CODE / 'prepare.py')}"

    for filename in (
        "execution_remote_h200.json",
        "execution_remote_fleet.json",
        "execution_remote_shared_single.json",
    ):
        execution_path = ROOT / "templates" / filename
        execution = json.loads(execution_path.read_text())
        code_rows = {row["execution_path"]: row for row in execution["code_bindings"]}
        assert set(code_rows) == {
            "baseline_provenance.json",
            "prepare.py",
            "pyproject.toml",
            "train.py",
            "uv.lock",
        }
        for row in code_rows.values():
            source = (execution_path.parent / row["source"]).resolve()
            assert source.is_file(), source
        assert execution["data_bindings"] == [
            {"source": "../runs/data_manifest.json", "execution_path": "data_manifest.json"}
        ]
