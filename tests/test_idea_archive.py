import unittest
from dataclasses import replace

from vibeautoresearch.core import SchemaError
from vibeautoresearch.idea_archive import IdeaRecord


def archived_idea() -> IdeaRecord:
    return IdeaRecord(
        idea_id="idea_archive_test",
        title="A literature-cleared systems mutation",
        version=1,
        summary="Remove redundant work while preserving the update.",
        experimental_plan="Run a paired screen and promote only through the funnel.",
        direction="systems_kernel",
        subsystem="attention_kernel",
        parent_idea_ids=(),
        scores={
            "interestingness": {"score": 8, "rationale": "Tests a real bottleneck."},
            "novelty": {"score": 7, "rationale": "Not tested by the closest paper."},
            "feasibility": {"score": 9, "rationale": "A small, reversible change."},
        },
        novelty_check={
            "provider": "semantic_scholar",
            "status": "passed",
            "query_rounds": [
                {
                    "query": "attention kernel redundant work language model training",
                    "result_paper_ids": ["pap_closest"],
                    "assessment": "Related implementation, different intervention.",
                }
            ],
            "closest_paper_ids": ["pap_closest"],
            "evidence_ids": ["evd_novelty_search"],
            "max_semantic_similarity": 0.42,
            "discard_threshold": 0.8,
            "assessment": "Below the frozen discard threshold.",
        },
        status="selected",
        hypothesis_id="hyp_archive_test",
        created_at="2026-07-24",
    )


class IdeaArchiveSchemaTest(unittest.TestCase):
    def test_selected_idea_binds_scores_plan_and_literature_check(self):
        payload = archived_idea().to_dict()

        self.assertEqual(IdeaRecord.from_dict(payload), archived_idea())

    def test_self_score_cannot_escape_the_one_to_ten_scale(self):
        scores = dict(archived_idea().scores)
        scores["novelty"] = {"score": 11, "rationale": "Overconfident."}

        with self.assertRaisesRegex(SchemaError, "<= 10"):
            replace(archived_idea(), scores=scores)

    def test_novelty_search_is_capped_at_ten_refinement_rounds(self):
        novelty = dict(archived_idea().novelty_check)
        novelty["query_rounds"] = [
            {
                "query": f"query {index}",
                "result_paper_ids": ["pap_closest"],
                "assessment": "Related.",
            }
            for index in range(11)
        ]

        with self.assertRaisesRegex(SchemaError, "at most ten"):
            replace(archived_idea(), novelty_check=novelty)

    def test_high_similarity_idea_cannot_be_marked_passed(self):
        novelty = dict(archived_idea().novelty_check)
        novelty["max_semantic_similarity"] = 0.9

        with self.assertRaisesRegex(SchemaError, "cannot pass"):
            replace(archived_idea(), novelty_check=novelty)

    def test_selected_idea_requires_structured_literature_evidence(self):
        novelty = dict(archived_idea().novelty_check)
        novelty["evidence_ids"] = []

        with self.assertRaisesRegex(SchemaError, "structured literature evidence"):
            replace(archived_idea(), novelty_check=novelty)


if __name__ == "__main__":
    unittest.main()


class ComputedNoveltyDedupTest(unittest.TestCase):
    """The novelty floor is COMPUTED, not only self-attested (closes the fork gap)."""

    @staticmethod
    def _idea(idea_id, text, *, threshold=0.8, status="selected", parents=()):
        from types import SimpleNamespace

        return SimpleNamespace(
            idea_id=idea_id,
            title=text,
            summary=text,
            experimental_plan=text,
            direction="optimizer",
            subsystem="sub",
            parent_idea_ids=parents,
            status=status,
            novelty_check={"discard_threshold": threshold},
        )

    def test_lexical_similarity_catches_identical_text(self):
        from vibeautoresearch.idea_archive import idea_tokens, lexical_similarity

        a = idea_tokens("raise the matrix learning rate for muon")
        b = idea_tokens("raise the matrix learning rate for muon")
        self.assertEqual(lexical_similarity(a, b), 1.0)
        c = idea_tokens("swap the attention kernel to flash-attention-3")
        self.assertLess(lexical_similarity(a, c), 0.3)

    def test_fork_by_near_duplicate_is_flagged(self):
        from vibeautoresearch.idea_archive import duplicate_ideas

        text = "raise the matrix learning rate to improve muon convergence"
        a = self._idea("idea_aaaaaaaa", text)
        b = self._idea("idea_bbbbbbbb", text)  # different id, self-reports nothing
        self.assertTrue(duplicate_ideas([a, b]))

    def test_parent_child_lineage_is_exempt(self):
        from vibeautoresearch.idea_archive import duplicate_ideas

        text = "raise the matrix learning rate to improve muon convergence"
        a = self._idea("idea_aaaaaaaa", text)
        child = self._idea("idea_cccccccc", text + " and add warmup", parents=("idea_aaaaaaaa",))
        self.assertFalse(duplicate_ideas([a, child]))

    def test_distinct_ideas_pass(self):
        from vibeautoresearch.idea_archive import duplicate_ideas

        a = self._idea("idea_aaaaaaaa", "raise the matrix learning rate for muon convergence")
        d = self._idea("idea_dddddddd", "switch attention to flash-attention-3 windowed sparsity")
        self.assertFalse(duplicate_ideas([a, d]))
