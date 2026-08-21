"""Fail-closed contracts for the packer-sourced FA3 boundary sidecar.

The local development host has no CUDA device, so stream/event execution is a
governed H200 diagnostic. These tests still exercise host invariants and lease
state transitions, and use AST/source contracts to prevent the scanner control
path or the backward-to-release ordering from drifting silently.
"""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

try:
    import torch
except ModuleNotFoundError as exc:
    raise unittest.SkipTest(
        "packer boundary host/behavior tests require PyTorch; the governed "
        "H200 suite executes them with the pinned runtime"
    ) from exc

import lib


ROOT = Path(__file__).resolve().parents[1]
LIB_SOURCE = (ROOT / "lib.py").read_text(encoding="utf-8")
TRAIN_SOURCE = (ROOT / "train.py").read_text(encoding="utf-8")
LIB_TREE = ast.parse(LIB_SOURCE)
TRAIN_TREE = ast.parse(TRAIN_SOURCE)


def _function_source(tree, source, name):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.get_source_segment(source, node)
    raise AssertionError(f"function {name!r} not found")


def _class_source(tree, source, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return ast.get_source_segment(source, node)
    raise AssertionError(f"class {name!r} not found")


class PackerBoundaryHostTest(unittest.TestCase):
    def test_boundary_invariants_accept_exact_flattened_rows(self):
        lib._validate_packer_boundaries([0, 3, 8, 11, 16], B=2, T=8)

    def test_boundary_invariants_reject_duplicate_or_missing_row_start(self):
        for boundaries in ([0, 3, 8, 8, 16], [0, 3, 9, 16]):
            with self.subTest(boundaries=boundaries):
                with self.assertRaisesRegex(
                    RuntimeError, "PACKER_DOC_BOUNDARY_INVARIANT_FAILED"
                ):
                    lib._validate_packer_boundaries(boundaries, B=2, T=8)

    def test_lease_fails_release_before_wait_and_double_release(self):
        class FakeStream:
            def __init__(self):
                self.waited = []
                self.cuda_stream = 123

            def wait_event(self, event):
                self.waited.append(event)

        class FakeEvent:
            def __init__(self):
                self.recorded = []

            def record(self, stream):
                self.recorded.append(stream)

        stream = FakeStream()
        slot = SimpleNamespace(
            cuda=torch.zeros(2, dtype=torch.int32),
            h2d_done=FakeEvent(),
            consumer_done=FakeEvent(),
            consumer_recorded=False,
            leased=True,
            generation=1,
        )
        lease = lib.BoundaryLease(slot, 1, 2, 8)
        with self.assertRaisesRegex(
            RuntimeError, "PACKER_BOUNDARY_LEASE_RELEASE_BEFORE_WAIT"
        ):
            with patch.object(lib.torch.cuda, "current_stream", return_value=stream):
                lease.mark_consumed()
        with patch.object(lib.torch.cuda, "current_stream", return_value=stream):
            view = lease.wait_for_current_stream()
        self.assertEqual(tuple(view.shape), (2,))
        with patch.object(lib.torch.cuda, "current_stream", return_value=stream):
            lease.mark_consumed()
        self.assertTrue(slot.consumer_recorded)
        self.assertFalse(slot.leased)
        with self.assertRaisesRegex(
            RuntimeError, "PACKER_BOUNDARY_LEASE_DOUBLE_RELEASE"
        ):
            with patch.object(lib.torch.cuda, "current_stream", return_value=stream):
                lease.mark_consumed()


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA loader buffers")
class PackerBoundaryBehaviorTest(unittest.TestCase):
    class FakeTokenizer:
        def __init__(self, bodies):
            self.bodies = bodies

        def get_bos_token_id(self):
            return 99

        def encode(self, doc_batch, prepend):
            return [
                [prepend, *self.bodies[document]]
                for document in doc_batch
            ]

    @staticmethod
    def _document_batches(documents):
        def factory(_split):
            epoch = 1
            while True:
                yield list(documents), epoch
                epoch += 1

        return factory

    @staticmethod
    def _scanner_boundaries(inputs, bos=99):
        flat = inputs.reshape(-1)
        starts = torch.nonzero(flat == bos, as_tuple=False).flatten()
        terminal = torch.tensor(
            [flat.numel()], device=flat.device, dtype=starts.dtype
        )
        return torch.cat((starts, terminal)).to(torch.int32)

    def _first_pair(self, documents, bodies, *, B, T):
        tokenizer = self.FakeTokenizer(bodies)
        factory = self._document_batches(documents)
        with patch.object(lib, "_document_batches", factory):
            legacy = lib.make_dataloader(
                tokenizer, B, T, "train", buffer_size=len(documents)
            )
            sidecar = lib.make_dataloader(
                tokenizer,
                B,
                T,
                "train",
                buffer_size=len(documents),
                return_doc_boundaries=True,
            )
            x_legacy, y_legacy, epoch_legacy = next(legacy)
            x_side, y_side, epoch_side, lease = next(sidecar)
            x_legacy = x_legacy.clone()
            y_legacy = y_legacy.clone()
            x_side = x_side.clone()
            y_side = y_side.clone()
            cu_side = lease.cu_seqlens.clone()
            lease.mark_consumed()
        torch.cuda.synchronize()
        return (
            x_legacy,
            y_legacy,
            epoch_legacy,
            x_side,
            y_side,
            epoch_side,
            cu_side,
        )

    def test_empty_document_start_at_pos_t_is_target_only(self):
        pair = self._first_pair(
            ["long", "empty", "medium", "short"],
            {
                "long": [1, 2, 3, 4, 5, 6, 7],
                "empty": [],
                "medium": [20, 21, 22],
                "short": [30, 31],
            },
            B=2,
            T=8,
        )
        (
            x_legacy,
            y_legacy,
            epoch_legacy,
            x_side,
            y_side,
            epoch_side,
            cu_side,
        ) = pair
        torch.testing.assert_close(x_legacy, x_side, rtol=0, atol=0)
        torch.testing.assert_close(y_legacy, y_side, rtol=0, atol=0)
        self.assertEqual(epoch_legacy, epoch_side)
        scanner = self._scanner_boundaries(x_legacy)
        torch.testing.assert_close(cu_side, scanner, rtol=0, atol=0)
        torch.testing.assert_close(
            cu_side.cpu(),
            torch.tensor([0, 8, 16], dtype=torch.int32),
            rtol=0,
            atol=0,
        )
        self.assertTrue(torch.all(y_legacy[:, -1] == 99).item())

    def test_best_fit_tie_crop_and_interior_bos_match_scanner(self):
        pair = self._first_pair(
            ["long_a", "long_b", "mid", "short"],
            {
                "long_a": [1, 2, 3, 4, 5, 6],
                "long_b": [11, 12, 13, 14, 15, 16],
                "mid": [21, 22, 23],
                "short": [31, 32],
            },
            B=1,
            T=8,
        )
        (
            x_legacy,
            y_legacy,
            epoch_legacy,
            x_side,
            y_side,
            epoch_side,
            cu_side,
        ) = pair
        torch.testing.assert_close(x_legacy, x_side, rtol=0, atol=0)
        torch.testing.assert_close(y_legacy, y_side, rtol=0, atol=0)
        self.assertEqual(epoch_legacy, epoch_side)
        self.assertEqual(
            x_side[0].cpu().tolist(),
            [99, 1, 2, 3, 4, 5, 6, 99],
        )
        self.assertEqual(
            y_side[0].cpu().tolist(),
            [1, 2, 3, 4, 5, 6, 99, 31],
        )
        scanner = self._scanner_boundaries(x_legacy)
        torch.testing.assert_close(cu_side, scanner, rtol=0, atol=0)
        torch.testing.assert_close(
            cu_side.cpu(),
            torch.tensor([0, 7, 8], dtype=torch.int32),
            rtol=0,
            atol=0,
        )


class PackerBoundarySourceContractTest(unittest.TestCase):
    def test_public_loader_flag_is_keyword_only_and_legacy_is_isolated(self):
        make_node = next(
            node
            for node in LIB_TREE.body
            if isinstance(node, ast.FunctionDef) and node.name == "make_dataloader"
        )
        self.assertEqual(
            [arg.arg for arg in make_node.args.kwonlyargs],
            ["return_doc_boundaries"],
        )
        public = ast.get_source_segment(LIB_SOURCE, make_node)
        legacy = _function_source(
            LIB_TREE, LIB_SOURCE, "_make_legacy_dataloader"
        )
        self.assertIn("return _make_legacy_dataloader", public)
        self.assertNotIn("BoundaryLease", legacy)
        self.assertNotIn("_BoundaryRing", legacy)
        self.assertIn("yield inputs, targets, epoch", legacy)
        self.assertNotIn("yield inputs, targets, epoch,", legacy)

    def test_sidecar_records_start_before_write_and_skips_pos_equal_t(self):
        sidecar = _function_source(
            LIB_TREE, LIB_SOURCE, "_make_boundary_dataloader"
        )
        record = sidecar.index("if pos < T:")
        best_fit_write = sidecar.index(
            "row_buffer[row_idx, pos:pos + len(doc)]"
        )
        crop_write = sidecar.index(
            "row_buffer[row_idx, pos:pos + remaining]"
        )
        self.assertLess(record, best_fit_write)
        self.assertLess(record, crop_write)
        self.assertIn("boundaries.append(row_idx * T + pos)", sidecar)
        self.assertIn("boundaries.append(B * T)", sidecar)
        self.assertIn("yield inputs, targets, epoch, boundary_lease", sidecar)

    def test_ring_is_fixed_three_slot_int32_pinned_cpu_and_cuda(self):
        slot = _class_source(LIB_TREE, LIB_SOURCE, "_BoundarySlot")
        ring = _class_source(LIB_TREE, LIB_SOURCE, "_BoundaryRing")
        self.assertIn("dtype=torch.int32, pin_memory=True", slot)
        self.assertIn("dtype=torch.int32, device=device", slot)
        self.assertEqual(slot.count("torch.cuda.Event(enable_timing=False)"), 2)
        self.assertIn("if slots != 3:", ring)
        self.assertIn("torch.cuda.Stream(device=device)", ring)
        self.assertNotIn(".record_stream(", LIB_SOURCE)

    def test_ring_has_separate_cpu_copy_and_cuda_consumer_lifetimes(self):
        stage = _function_source(LIB_TREE, LIB_SOURCE, "stage")
        sync = stage.index("slot.h2d_done.synchronize()")
        cpu_write = stage.index("slot.cpu[:count].copy_")
        consumer_wait = stage.index("self._copy_stream.wait_event(slot.consumer_done)")
        cuda_copy = stage.index("slot.cuda[:count].copy_")
        h2d_record = stage.index("slot.h2d_done.record(self._copy_stream)")
        self.assertLess(sync, cpu_write)
        self.assertLess(consumer_wait, cuda_copy)
        self.assertLess(cuda_copy, h2d_record)
        self.assertIn(
            "PACKER_BOUNDARY_RING_REUSE_WITH_OUTSTANDING_LEASE", stage
        )

    def test_training_stream_wait_and_backward_release_order(self):
        lease = _class_source(LIB_TREE, LIB_SOURCE, "BoundaryLease")
        self.assertIn("stream.wait_event(self._slot.h2d_done)", lease)
        self.assertIn("self._slot.consumer_done.record(stream)", lease)
        self.assertIn("PACKER_BOUNDARY_LEASE_DOUBLE_RELEASE", lease)

        loop_start = TRAIN_SOURCE.index("while True:\n    OBS.clear()")
        loop_end = TRAIN_SOURCE.index("print()  # newline after", loop_start)
        loop = TRAIN_SOURCE[loop_start:loop_end]
        backward = loop.index("loss.backward()")
        release = loop.index("boundary_lease.mark_consumed()")
        prefetch = loop.index(
            "x, y, epoch, boundary_lease = next(train_loader)", release
        )
        self.assertLess(backward, release)
        self.assertLess(release, prefetch)

    def test_sidecar_payload_is_outside_the_single_compiled_graph(self):
        builder = _function_source(
            TRAIN_TREE, TRAIN_SOURCE, "build_doc_masks_from_boundaries"
        )
        self.assertIn("cu = boundary_lease.cu_seqlens", builder)
        self.assertIn("torch._dynamo.mark_dynamic(cu, 0)", builder)
        self.assertIn("_DOC_MASK_BOUNDARY_COUNTS.append", builder)
        self.assertEqual(TRAIN_SOURCE.count("model = torch.compile("), 1)

    def test_flags_fail_closed_and_are_echoed(self):
        self.assertIn(
            'if _packer_doc_boundaries_raw not in {"0", "1"}:', TRAIN_SOURCE
        )
        self.assertIn(
            'if PACKER_BOUNDARY_VERIFY_BATCHES not in {0, 1000}:',
            TRAIN_SOURCE,
        )
        self.assertIn(
            '"PACKER_DOC_BOUNDARIES=1 requires DOC_MASK=1 "', TRAIN_SOURCE
        )
        self.assertIn(
            "PACKER_DOC_BOUNDARIES={PACKER_DOC_BOUNDARIES}", TRAIN_SOURCE
        )
        self.assertIn(
            "PACKER_BOUNDARY_VERIFY_BATCHES={PACKER_BOUNDARY_VERIFY_BATCHES}",
            TRAIN_SOURCE,
        )
        self.assertIn(
            "PACKER_DIAGNOSTIC_HASHES={PACKER_DIAGNOSTIC_HASHES}",
            TRAIN_SOURCE,
        )
        self.assertIn(
            "PACKER_SIDECAR_ACTIVATE_STEP={PACKER_SIDECAR_ACTIVATE_STEP}",
            TRAIN_SOURCE,
        )
        self.assertIn(
            "PACKER_SIDECAR_ACTIVATE_STEP != 20",
            TRAIN_SOURCE,
        )

    def test_preclock_gate_uses_independent_loaders_and_exact_1000_batches(self):
        verifier = _function_source(
            TRAIN_TREE, TRAIN_SOURCE, "verify_packer_boundary_equivalence"
        )
        self.assertIn("if batches != 1000:", verifier)
        self.assertIn("scanner_loader = make_dataloader", verifier)
        self.assertIn("sidecar_loader = make_dataloader", verifier)
        self.assertIn("return_doc_boundaries=True", verifier)
        self.assertIn("torch.equal(x_scan, x_side)", verifier)
        self.assertIn("torch.equal(y_scan, y_side)", verifier)
        self.assertIn("torch.equal(scanner_cpu, sidecar_cpu)", verifier)
        self.assertIn("PACKER_BOUNDARY_INTERIOR_BOS_MISMATCH", verifier)
        self.assertIn("PACKER_BOUNDARY_VERIFY_PASSED=1", verifier)
        self.assertIn("data_sha256=", verifier)
        self.assertIn("boundary_sha256=", verifier)
        verification_call = TRAIN_SOURCE.index(
            "if PACKER_BOUNDARY_VERIFY_BATCHES:"
        )
        training_clock = TRAIN_SOURCE.index("t_start_training = time.time()")
        self.assertLess(verification_call, training_clock)

    def test_scanner_remains_control_and_all_evaluation_paths(self):
        self.assertIn(
            "build_doc_masks_for_batch(\n"
            "                x, WINDOW_LEFTS, record_boundary_count=True",
            TRAIN_SOURCE,
        )
        self.assertIn(
            "doc_masks=build_doc_masks_for_batch(x_val, WINDOW_LEFTS)",
            TRAIN_SOURCE,
        )
        self.assertIn(
            "doc_masks=build_doc_masks_for_batch(_x, WINDOW_LEFTS)",
            TRAIN_SOURCE,
        )
        self.assertIn(
            "forward_kwargs_fn=_doc_mask_forward_kwargs",
            TRAIN_SOURCE,
        )

    def test_diagnostic_contract_has_aa_activation_graph_and_clean_profile(self):
        self.assertIn("PACKER_DIAGNOSTIC_AA_ATTESTED=1", TRAIN_SOURCE)
        self.assertIn("PACKER_DIAGNOSTIC_AB_ATTESTED=1", TRAIN_SOURCE)
        self.assertIn("PACKER_SIDECAR_PHASE_ACTIVATED=1", TRAIN_SOURCE)
        self.assertIn("PACKER_DIAGNOSTIC_GRAPH_ATTESTED=1", TRAIN_SOURCE)
        self.assertIn("PACKER_DIAGNOSTIC_PROFILE_ATTESTED=1", TRAIN_SOURCE)
        self.assertIn("PACKER_DIAGNOSTIC_PROFILE_MS=", TRAIN_SOURCE)
        self.assertIn(
            "step < PACKER_SIDECAR_ACTIVATE_STEP",
            TRAIN_SOURCE,
        )
        self.assertIn(
            "step >= PACKER_SIDECAR_ACTIVATE_STEP",
            TRAIN_SOURCE,
        )
        self.assertIn("boundary_lease.discard_unconsumed()", TRAIN_SOURCE)
        self.assertIn("diagnostic_model_sha256 = _canonical_state_sha256", TRAIN_SOURCE)
        self.assertIn(
            "diagnostic_optimizer_sha256 = _canonical_state_sha256", TRAIN_SOURCE
        )
        marker = TRAIN_SOURCE.index("PACKER_DIAGNOSTIC_HASHES_VERIFIED=1")
        for field in (
            "model_sha256=",
            "optimizer_sha256=",
            "aa_loss_sha256=",
            "aa_boundary_sha256=",
            "ab_loss_sha256=",
            "ab_data_sha256=",
            "ab_boundary_sha256=",
            "tail_loss_sha256=",
            "aa_data_sha256=",
            "scanner_micro_batches=",
            "sidecar_micro_batches=",
        ):
            self.assertIn(field, TRAIN_SOURCE[marker:marker + 900])


if __name__ == "__main__":
    unittest.main()
