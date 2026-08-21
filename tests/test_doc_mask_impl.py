"""Contracts for the DOC_MASK execution paths (dense vs block-sparse flex).

train.py cannot be imported without CUDA, so these are source/AST contracts —
the same style as the other spec tests. Numerical equivalence between the two
implementations is checked on GPU by tools/bench_doc_mask_attention.py.
"""

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _function_source(tree, source, name, cls=None):
    """Source text of a (optionally class-scoped) function definition."""
    for node in ast.walk(tree):
        if cls is not None:
            if not (isinstance(node, ast.ClassDef) and node.name == cls):
                continue
            scope = node.body
        else:
            scope = [node]
        for child in scope:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == name:
                return ast.get_source_segment(source, child)
    return None


class DocMaskImplTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.src = (ROOT / "train.py").read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.src)
        cls.lib_src = (ROOT / "lib.py").read_text(encoding="utf-8")

    def test_impl_env_is_validated_and_defaults_to_varlen(self):
        # ADOPTED 2026-07-28 (paired n=10, mean -0.005222, 10/10 same sign): the
        # fa3 varlen cu_seqlens path replaced dense as the default. Same masking
        # semantics, different kernel -- dense materializes a (B,1,T,T) mask that
        # is re-read every layer in fwd and bwd, varlen lets the kernel skip
        # cross-document blocks entirely.
        self.assertIn('DOC_MASK_IMPL = os.environ.get("DOC_MASK_IMPL", "varlen").lower()', self.src)
        self.assertIn('"dense", "flex", "varlen"', self.src)

    def test_rejected_static_cu_seqlens_path_is_not_executable(self):
        # gap_fa3_varlen_static_cu_seqlens: the static-shape construction is
        # byte-identical on cu_seqlens content (verified over 200 randomized
        # trials) but fa3 varlen REJECTS it at runtime with an illegal memory
        # access, because it infers its sequence count from cu_seqlens.numel()-1.
        # A known crash-on-enable branch must not remain in the training surface.
        self.assertNotIn("DOC_MASK_STATIC_CU", self.src)
        self.assertNotIn("DOC_MASK_CU_CAPACITY", self.src)

    def test_flex_requires_the_sdpa_backend(self):
        # fa3/fa4 never take the doc-mask path; silently ignoring the request
        # would report a flex run that actually ran unmasked attention.
        self.assertIn('if DOC_MASK_IMPL == "flex" and ATTN_BACKEND != "sdpa":', self.src)

    def test_masks_are_built_once_per_forward_not_once_per_layer(self):
        """The P1 this path exists to fix.

        A (B, 1, T, T) bool mask at B=72, T=2048 is 302 MB. Building it inside
        CausalSelfAttention.forward meant one per layer per micro-step; the
        builders must be reachable only from GPT.forward / the batch helper.
        """
        attn_forward = _function_source(self.tree, self.src, "forward", cls="CausalSelfAttention")
        self.assertIsNotNone(attn_forward)
        self.assertNotIn("_build_dense_doc_mask", attn_forward)
        self.assertNotIn("_build_flex_doc_mask", attn_forward)
        self.assertIn("_doc_masked_attention(q, k, v, doc_mask)", attn_forward)

        gpt_forward = _function_source(self.tree, self.src, "forward", cls="GPT")
        self.assertIsNotNone(gpt_forward)
        self.assertIn("_build_dense_doc_mask", gpt_forward)
        # Deduplicated by window key, so DOC_MASK_MODE=seg builds exactly one.
        self.assertIn("key = _doc_mask_key(window_size[0])", gpt_forward)
        self.assertIn("if key not in doc_masks:", gpt_forward)

    def test_seg_mode_shares_a_single_mask_across_layers(self):
        key_fn = _function_source(self.tree, self.src, "_doc_mask_key")
        self.assertIsNotNone(key_fn)
        self.assertIn('if DOC_MASK_MODE == "seg":', key_fn)
        self.assertIn("return -1", key_fn)

    def test_flex_fails_closed_when_masks_were_not_prebuilt(self):
        """create_block_mask cannot be traced inside fullgraph=True.

        Falling back to dense would silently run the arm the experiment is
        supposed to be measuring against, at 4x the attention cost.
        """
        gpt_forward = _function_source(self.tree, self.src, "forward", cls="GPT")
        self.assertIn('assert DOC_MASK_IMPL != "flex"', gpt_forward)

    def test_batch_mask_builder_is_inert_off_the_flex_path(self):
        builder = _function_source(self.tree, self.src, "build_doc_masks_for_batch")
        self.assertIsNotNone(builder)
        self.assertIn(
            'if not DOC_MASK or ATTN_BACKEND != "sdpa" or DOC_MASK_IMPL != "flex":',
            builder,
        )
        self.assertIn("return None", builder)

    def test_mask_build_is_charged_to_the_timed_region(self):
        # If the per-step BlockMask build were hoisted out of the timed loop the
        # flex arm would under-report its own cost in every Track B comparison.
        scanner = self.src.index(
            "build_doc_masks_for_batch(\n"
            "                x, WINDOW_LEFTS, record_boundary_count=True\n"
            "            )"
        )
        forward = self.src.index("with autocast_ctx:", scanner)
        self.assertLess(scanner, forward)
        self.assertIn(
            "build_doc_masks_from_boundaries(\n"
            "                boundary_lease, WINDOW_LEFTS, record_boundary_count=True\n"
            "            )",
            self.src,
        )
        # The property under test is that the loss call inside the timed region
        # passes the freshly-built doc_masks, so the mask build is charged. The
        # TARGET argument was renamed y -> _y when OFFSET_AUG was added (it is
        # `_offset_targets(y, ...) if OFFSET_AUG else y`), which does not affect
        # what is being asserted. Match the input and the doc_masks kwarg, and
        # require exactly one such call so a second, untimed one cannot appear.
        calls = [
            line.strip()
            for line in self.src.splitlines()
            if line.strip().startswith("loss = model(x,")
            and "doc_masks=_doc_masks" in line
        ]
        self.assertEqual(calls, ["loss = model(x, _y, doc_masks=_doc_masks)"])

    def test_compact_boundaries_have_a_symmetric_fail_closed_runtime_attestation(self):
        helper = _function_source(self.tree, self.src, "_compact_doc_boundaries")
        self.assertIsNotNone(helper)
        self.assertIn("starts[:, 0] = True", helper)
        self.assertIn("torch.nonzero(flat", helper)
        self.assertIn("dtype=torch.int32", helper)
        self.assertIn("torch._dynamo.mark_dynamic(cu, 0)", helper)

        attestation = _function_source(
            self.tree, self.src, "attest_compact_doc_boundaries"
        )
        self.assertIsNotNone(attestation)
        for invariant in (
            "first_zero",
            "last_equals_b_times_t",
            "strictly_increasing",
            "all_row_starts_present",
            "max_gap_lte_t",
        ):
            self.assertIn(invariant, attestation)
        self.assertIn("DOC_MASK_BOUNDARY_ATTESTATION_FAILED", attestation)
        self.assertIn("DOC_MASK_BOUNDARY_ATTESTED=1", attestation)

        # The pre-clock attestation is deliberately keyed only to the common
        # backend/implementation, not DOC_MASK, so explicit off/on arms both pay
        # the same setup/warmup work.
        self.assertIn(
            'if ATTN_BACKEND == "fa3" and DOC_MASK_IMPL == "varlen":\n'
            "    # Symmetric, pre-clock, fail-closed attestation for both explicit arms.\n"
            "    attest_compact_doc_boundaries(x)",
            self.src,
        )

    def test_every_treatment_training_batch_records_its_boundary_count(self):
        self.assertIn("_DOC_MASK_BOUNDARY_COUNTS = []", self.src)
        self.assertIn(
            "_DOC_MASK_BOUNDARY_COUNTS.append(cu.shape[0] - 1)", self.src
        )
        self.assertIn(
            "expected_boundary_batches = step * grad_accum_steps", self.src
        )
        self.assertIn("DOC_MASK_BOUNDARY_COUNTS_VERIFIED=1", self.src)
        self.assertIn("DOC_MASK_BOUNDARY_COUNTS=", self.src)

    def test_window_lefts_captured_before_compile_wrapper(self):
        capture = self.src.index("WINDOW_LEFTS = [w[0] for w in model.window_sizes]")
        compile_call = self.src.index("model = torch.compile(model, dynamic=False")
        self.assertLess(capture, compile_call)

    def test_every_model_entry_point_supplies_masks(self):
        """A missed call site only fails at the END of a 2000-step run.

        The final evaluate_bpb is the expensive one to discover this way, so
        each model-invoking path must thread the doc-mask kwargs.
        """
        self.assertIn("forward_kwargs_fn=_doc_mask_forward_kwargs", self.src)
        self.assertIn("def evaluate_bpb(model, tokenizer, batch_size, eval_tokens=None, forward_kwargs_fn=None)", self.lib_src)
        self.assertIn("extra = forward_kwargs_fn(x) if forward_kwargs_fn is not None else {}", self.lib_src)
        self.assertIn("loss_flat = model(x, y, reduction='none', **extra).view(-1)", self.lib_src)
        # Both eval entry points in train.py, plus the DIAGNOSE sweep.
        self.assertEqual(self.src.count("_doc_mask_forward_kwargs"), 3)
        self.assertIn("doc_masks=build_doc_masks_for_batch(x_val, WINDOW_LEFTS)", self.src)
        self.assertIn("doc_masks=build_doc_masks_for_batch(_x, WINDOW_LEFTS)", self.src)

    def test_impl_is_recorded_in_provenance(self):
        # A result artifact that does not name the kernel cannot be compared
        # against one that used the other kernel.
        self.assertIn('"DOC_MASK_IMPL": DOC_MASK_IMPL,', self.src)
        self.assertIn("DOC_MASK_IMPL={DOC_MASK_IMPL}", self.src)

    def test_both_impls_share_one_causal_window_envelope(self):
        # Divergent envelopes between impls would make the arms incomparable.
        envelope = _function_source(self.tree, self.src, "_doc_mask_causal_window")
        self.assertIsNotNone(envelope)
        dense = _function_source(self.tree, self.src, "_build_dense_doc_mask")
        self.assertIn("_doc_mask_causal_window(T, window_left, seg.device)", dense)


if __name__ == "__main__":
    unittest.main()
