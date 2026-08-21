"""Focused guards for Paper-020 Round-1 GPAS and its diagnostic contract."""

from __future__ import annotations

import ast
import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN = (ROOT / "train.py").read_text(encoding="utf-8")
TREE = ast.parse(TRAIN)


def _class(name: str) -> ast.ClassDef:
    return next(
        node for node in TREE.body if isinstance(node, ast.ClassDef) and node.name == name
    )


def _method(cls: ast.ClassDef, name: str) -> ast.FunctionDef:
    return next(
        node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _load_diagnostic_module():
    path = ROOT / "tools" / "gpas_mechanism_diagnostic.py"
    spec = importlib.util.spec_from_file_location("gpas_mechanism_diagnostic", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class GPASModelContractTest(unittest.TestCase):
    def test_flag_is_strict_and_default_off(self):
        self.assertIn('os.environ.get("GPAS_ENABLE", "0")', TRAIN)
        self.assertIn('_gpas_enable_raw not in {"0", "1"}', TRAIN)
        self.assertIn("GPAS_ENABLE = _gpas_enable_raw == \"1\"", TRAIN)
        self.assertIn("gpas=GPAS_ENABLE", TRAIN)

    def test_one_zero_init_scalar_is_shared_after_both_residual_sums(self):
        block = _class("Block")
        init_source = ast.get_source_segment(TRAIN, _method(block, "__init__"))
        forward_source = ast.get_source_segment(TRAIN, _method(block, "forward"))
        self.assertEqual(
            init_source.count("self.gpas_alpha = nn.Parameter(torch.zeros(()))"), 1
        )
        exact = "x = x - F.silu(self.gpas_alpha) * x.detach()"
        self.assertEqual(forward_source.count(exact), 2)
        # The attention residual add is written either as the plain form or, once
        # PERI_LN_ATTN was added, as a flag-selected form. The property under test
        # is unchanged and still strict: there is EXACTLY ONE attention residual
        # add and it precedes the first GPAS application.
        attn_adds = [
            "x = x + self.attn(",
            "x = x + (norm(_attn_out) if PERI_LN_ATTN else _attn_out)",
        ]
        present = [form for form in attn_adds if form in forward_source]
        self.assertEqual(
            len(present), 1, f"expected exactly one attention residual form, got {present}"
        )
        self.assertEqual(forward_source.count(present[0]), 1)
        self.assertLess(
            forward_source.index(present[0]),
            forward_source.index(exact),
        )
        self.assertLess(
            forward_source.index("x = x + norm(self.mlp(norm(x)))"),
            forward_source.rindex(exact),
        )

    def test_meta_materialization_rezeros_without_rng(self):
        source = ast.get_source_segment(TRAIN, _method(_class("GPT"), "init_weights"))
        self.assertIn('if hasattr(block, "gpas_alpha"):', source)
        self.assertIn("torch.nn.init.zeros_(block.gpas_alpha)", source)
        self.assertNotIn("normal_(block.gpas_alpha", source)
        self.assertNotIn("uniform_(block.gpas_alpha", source)

    def test_optimizer_group_is_exact_and_schedule_exempt(self):
        source = ast.get_source_segment(TRAIN, _method(_class("GPT"), "setup_optimizer"))
        for marker in (
            '"params": gpas_params',
            '"lr": 0.005',
            '"betas": (0.8, 0.95)',
            '"eps": 1e-10',
            '"weight_decay": 0.0',
            '"is_gpas": True',
        ):
            self.assertIn(marker, source)
        self.assertIn('elif group.get("is_gpas", False):', TRAIN)
        self.assertIn("gpas_groups.append(group)", TRAIN)
        # The default-off control must not retain even an empty GPAS optimizer
        # group: the common empty-group filter runs before optimizer creation.
        empty_filter = 'param_groups = [g for g in param_groups if g["params"]]'
        self.assertIn(empty_filter, source)
        self.assertLess(
            source.index(empty_filter),
            source.index("optimizer = MuonAdamW(param_groups)"),
        )
        # Only adam_group_lrs are warmup/warmdown scheduled; GPAS is excluded.
        self.assertNotIn("gpas_group_lrs", TRAIN)
        self.assertNotIn('"demon_beta1": True,\n                "is_gpas"', TRAIN)

    def test_diagnostic_variance_and_gate_probes_are_eager_only(self):
        block_forward = ast.get_source_segment(
            TRAIN, _method(_class("Block"), "forward")
        )
        self.assertIn(".double().var(dim=-1, correction=0).mean()", block_forward)
        self.assertIn("post_mlp_residual_variance", block_forward)
        self.assertIn("if OBSERVE_LAYER_PROBES and GPAS_ENABLE:", TRAIN)
        self.assertIn("gpas_alpha_grad", TRAIN)
        self.assertIn("torch.compile disabled for Python probe collection", TRAIN)


class GPASDiagnosticContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.diag = _load_diagnostic_module()

    def test_packer_flags_must_all_be_zero(self):
        zero = {
            key: "0" for key in self.diag.PACKER_ZERO_CONTRACT
        }
        self.assertEqual(self.diag.require_packer_flags_zero(zero), zero)
        zero["PACKER_DIAGNOSTIC_HASHES"] = "1"
        with self.assertRaisesRegex(
            RuntimeError, "GPAS_DIAGNOSTIC_PACKER_FLAGS_NONZERO"
        ):
            self.diag.require_packer_flags_zero(zero)

    def test_exact_parity_comparison_accepts_only_complete_match(self):
        base = {
            field: f"same-{field}" for field in self.diag.PARITY_EXACT_FIELDS
        }
        base.update(
            {
                "packer_flags": dict(self.diag.PACKER_ZERO_CONTRACT),
                "compile_identity": {
                    "dynamo_graph_count": 1,
                    "dynamo_recompile_count": 0,
                },
                "gpas_gates": [],
            }
        )
        treatment = dict(base)
        treatment["gpas_gates"] = [
            {
                "alpha": 0.0,
                "gradient": 0.25,
                "forward_scale": 1.0,
            }
        ]
        self.assertEqual(
            self.diag.compare_step0_parity(base, treatment)["verdict"], "PASS"
        )
        treatment["loss_sha256"] = "different"
        with self.assertRaisesRegex(RuntimeError, "GPAS_STEP0_PARITY_FAILED"):
            self.diag.compare_step0_parity(base, treatment)


if __name__ == "__main__":
    unittest.main()
