import unittest
from dataclasses import replace

from vibeautoresearch.core import SchemaError
from vibeautoresearch.evidence import resolve_terminal_evidence
from vibeautoresearch.knowledge import EvidenceRecord
from vibeautoresearch.registry import ResearchRegistry


def literature_evidence(
    evidence_id: str,
    *,
    paper_ids: tuple[str, ...] = ("pap_subject",),
    claim_ids: tuple[str, ...] = ("clm_subject",),
    hypothesis_ids: tuple[str, ...] = (),
    relation: str = "supports",
    strength: str = "moderate",
    supersedes: str = "",
) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=evidence_id,
        source_type="literature",
        paper_ids=paper_ids,
        claim_ids=claim_ids,
        run_ids=(),
        experiment_id="",
        hypothesis_ids=hypothesis_ids,
        facts={"reported": evidence_id},
        analysis={"method": "primary_read", "code_ref": "paper"},
        trust={
            "design": "adequate",
            "replication": "weak",
            "scope_match": "adequate",
            "directness": "strong",
            "limitations": [],
        },
        assessment={
            "relation": relation,
            "strength": strength,
            "limitations": [],
        },
        artifact_paths=(),
        created_at="2026-07-29T00:00:00Z",
        created_by="test",
        supersedes_evidence_id=supersedes,
    )


def run_evidence(
    evidence_id: str,
    *,
    run_ids: tuple[str, ...] = ("run_subject_1",),
    experiment_id: str = "exp_subject",
    hypothesis_ids: tuple[str, ...] = ("hyp_subject",),
    supersedes: str = "",
) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=evidence_id,
        source_type="internal_run",
        paper_ids=(),
        claim_ids=(),
        run_ids=run_ids,
        experiment_id=experiment_id,
        hypothesis_ids=hypothesis_ids,
        facts={"reported": evidence_id},
        analysis={"method": "paired_endpoint", "code_ref": "tests"},
        trust={
            "design": "strong",
            "replication": "adequate",
            "scope_match": "strong",
            "directness": "strong",
            "limitations": [],
        },
        assessment={
            "relation": "supports",
            "strength": "moderate",
            "limitations": [],
        },
        artifact_paths=(),
        created_at="2026-07-29T00:00:00Z",
        created_by="test",
        supersedes_evidence_id=supersedes,
    )


class TerminalEvidenceResolverTest(unittest.TestCase):
    def test_unique_chain_resolves_every_historical_id_to_one_terminal(self):
        first = literature_evidence("evd_chain_first")
        second = literature_evidence(
            "evd_chain_second", supersedes=first.evidence_id
        )
        terminal = literature_evidence(
            "evd_chain_terminal",
            relation="inconclusive",
            strength="weak",
            supersedes=second.evidence_id,
        )

        index = resolve_terminal_evidence(
            {
                first.evidence_id: first,
                second.evidence_id: second,
                terminal.evidence_id: terminal,
            }
        )

        self.assertEqual(set(index.history), {
            first.evidence_id,
            second.evidence_id,
            terminal.evidence_id,
        })
        self.assertEqual(set(index.terminals), {terminal.evidence_id})
        self.assertIs(index.terminal_record(first.evidence_id), terminal)
        self.assertEqual(
            index.terminal_records(
                (first.evidence_id, second.evidence_id, terminal.evidence_id)
            ),
            (terminal,),
        )

    def test_self_supersession_is_rejected(self):
        item = literature_evidence(
            "evd_self", supersedes="evd_self"
        )
        with self.assertRaisesRegex(SchemaError, "cannot supersede itself"):
            resolve_terminal_evidence({item.evidence_id: item})

    def test_missing_predecessor_is_rejected(self):
        item = literature_evidence(
            "evd_missing_successor", supersedes="evd_missing_predecessor"
        )
        with self.assertRaisesRegex(SchemaError, "supersedes missing evidence"):
            resolve_terminal_evidence({item.evidence_id: item})

    def test_cycle_is_rejected(self):
        first = literature_evidence(
            "evd_cycle_first", supersedes="evd_cycle_second"
        )
        second = literature_evidence(
            "evd_cycle_second", supersedes="evd_cycle_first"
        )
        with self.assertRaisesRegex(SchemaError, "supersession cycle"):
            resolve_terminal_evidence(
                {first.evidence_id: first, second.evidence_id: second}
            )

    def test_branch_is_rejected_to_preserve_unique_successor(self):
        predecessor = literature_evidence("evd_branch_root")
        left = literature_evidence(
            "evd_branch_left", supersedes=predecessor.evidence_id
        )
        right = literature_evidence(
            "evd_branch_right", supersedes=predecessor.evidence_id
        )
        with self.assertRaisesRegex(SchemaError, "multiple successors"):
            resolve_terminal_evidence(
                {
                    predecessor.evidence_id: predecessor,
                    left.evidence_id: left,
                    right.evidence_id: right,
                }
            )

    def test_source_type_change_is_rejected(self):
        predecessor = literature_evidence("evd_source_root")
        successor = replace(
            run_evidence("evd_source_changed"),
            supersedes_evidence_id=predecessor.evidence_id,
        )
        with self.assertRaisesRegex(SchemaError, "changes source_type"):
            resolve_terminal_evidence(
                {
                    predecessor.evidence_id: predecessor,
                    successor.evidence_id: successor,
                }
            )

    def test_operational_successor_may_add_but_not_drop_anchors(self):
        predecessor = run_evidence("evd_run_root")
        extended = run_evidence(
            "evd_run_extended",
            run_ids=("run_subject_1", "run_subject_2"),
            hypothesis_ids=("hyp_subject", "hyp_replication"),
            supersedes=predecessor.evidence_id,
        )
        index = resolve_terminal_evidence(
            {
                predecessor.evidence_id: predecessor,
                extended.evidence_id: extended,
            }
        )
        self.assertIs(index.terminal_record(predecessor.evidence_id), extended)

        dropped = run_evidence(
            "evd_run_dropped",
            run_ids=("run_subject_2",),
            supersedes=predecessor.evidence_id,
        )
        with self.assertRaisesRegex(SchemaError, "changes operational subject"):
            resolve_terminal_evidence(
                {
                    predecessor.evidence_id: predecessor,
                    dropped.evidence_id: dropped,
                }
            )

    def test_fa4_new_atomic_claim_is_not_a_valid_supersession_subject(self):
        old = literature_evidence(
            "evd_lit_fa4_hopper_scope_mismatch",
            paper_ids=("pap_flashattention4",),
            claim_ids=("clm_fa4_blackwell_hardware_coupling",),
        )
        scope_correction = literature_evidence(
            "evd_lit_fa4_hopper_scope_correction",
            paper_ids=("pap_flashattention4",),
            claim_ids=("clm_fa4_blackwell_hardware_coupling",),
            relation="mixed",
            supersedes=old.evidence_id,
        )
        valid_index = resolve_terminal_evidence(
            {
                old.evidence_id: old,
                scope_correction.evidence_id: scope_correction,
            }
        )
        self.assertIs(
            valid_index.terminal_record(old.evidence_id), scope_correction
        )

        primary = literature_evidence(
            "evd_lit_fa4_h200_lpt_scheduler",
            paper_ids=("pap_flashattention4_primary",),
            claim_ids=("clm_fa4_h200_lpt_scheduler",),
            supersedes=old.evidence_id,
        )
        separate = replace(primary, supersedes_evidence_id="")
        separate_index = resolve_terminal_evidence(
            {old.evidence_id: old, separate.evidence_id: separate}
        )
        self.assertEqual(
            set(separate_index.terminals),
            {old.evidence_id, separate.evidence_id},
        )
        with self.assertRaisesRegex(SchemaError, "claim_ids were dropped"):
            resolve_terminal_evidence(
                {old.evidence_id: old, primary.evidence_id: primary}
            )

    def test_tensorizing_engram_new_mechanism_is_separate_evidence(self):
        old = literature_evidence(
            "evd_lit_tensorizing_engram",
            paper_ids=("pap_tensorizing_engram",),
            claim_ids=("clm_engram_tensorizing",),
            hypothesis_ids=("hyp_c2_composite_codes",),
        )
        scope_correction = literature_evidence(
            "evd_lit_tensorizing_engram_scope_correction",
            paper_ids=("pap_tensorizing_engram",),
            claim_ids=("clm_engram_tensorizing",),
            hypothesis_ids=("hyp_c2_composite_codes",),
            relation="mixed",
            supersedes=old.evidence_id,
        )
        valid_index = resolve_terminal_evidence(
            {
                old.evidence_id: old,
                scope_correction.evidence_id: scope_correction,
            }
        )
        self.assertIs(
            valid_index.terminal_record(old.evidence_id), scope_correction
        )
        primary = literature_evidence(
            "evd_lit_tngram_cp_shared_factors_mechanism",
            paper_ids=("pap_tensorizing_engram_primary",),
            claim_ids=("clm_tngram_cp_shared_factors_mechanism",),
            hypothesis_ids=("hyp_c2_composite_codes",),
            supersedes=old.evidence_id,
        )
        separate = replace(primary, supersedes_evidence_id="")
        separate_index = resolve_terminal_evidence(
            {old.evidence_id: old, separate.evidence_id: separate}
        )
        self.assertEqual(
            set(separate_index.terminals),
            {old.evidence_id, separate.evidence_id},
        )
        with self.assertRaisesRegex(SchemaError, "claim_ids were dropped"):
            resolve_terminal_evidence(
                {old.evidence_id: old, primary.evidence_id: primary}
            )

    def test_ngram_scope_downgrade_is_same_subject_and_replaces_conclusion(self):
        original = literature_evidence(
            "evd_lit_ngram_regularizer_corroborates_family",
            paper_ids=("pap_ngram_aware_memorization_regularizer",),
            claim_ids=("clm_ngram_regularizer_reduces_memorization",),
            hypothesis_ids=("hyp_ngram_transfer",),
        )
        correction = literature_evidence(
            "evd_lit_ngram_regularizer_scope_correction",
            paper_ids=("pap_ngram_aware_memorization_regularizer",),
            claim_ids=("clm_ngram_regularizer_reduces_memorization",),
            hypothesis_ids=("hyp_ngram_transfer",),
            relation="not_tested",
            strength="weak",
            supersedes=original.evidence_id,
        )
        evidence = {
            original.evidence_id: original,
            correction.evidence_id: correction,
        }

        index = resolve_terminal_evidence(evidence)

        self.assertIs(index.terminal_record(original.evidence_id), correction)
        self.assertEqual(
            ResearchRegistry._conclusively_tested_hypothesis_ids(evidence),
            set(),
        )

    def test_exact_kimi_legacy_edge_is_grandfathered_only(self):
        secondary = literature_evidence(
            "evd_lit_kimi_k3",
            paper_ids=("pap_kimi_k3",),
            claim_ids=("clm_kda_linear_full_ratio",),
        )
        primary = literature_evidence(
            "evd_lit_kimi_k3_primary",
            paper_ids=("pap_kimi_k3_primary",),
            claim_ids=("clm_kda_mechanism",),
            supersedes=secondary.evidence_id,
        )
        index = resolve_terminal_evidence(
            {secondary.evidence_id: secondary, primary.evidence_id: primary}
        )
        self.assertIs(index.terminal_record(secondary.evidence_id), primary)

        imitation = literature_evidence(
            "evd_lit_kimi_k3_primary_copy",
            paper_ids=("pap_kimi_k3_primary",),
            claim_ids=("clm_kda_mechanism",),
            supersedes=secondary.evidence_id,
        )
        with self.assertRaisesRegex(SchemaError, "claim_ids were dropped"):
            resolve_terminal_evidence(
                {
                    secondary.evidence_id: secondary,
                    imitation.evidence_id: imitation,
                }
            )


if __name__ == "__main__":
    unittest.main()
