import json
from pathlib import Path
import unittest

from tools import register_flash_attention_3_varlen_literature as fa3


class FlashAttention3VarlenLiteratureRegistrationTests(unittest.TestCase):
    def test_typed_ids_and_exact_commit_are_frozen(self):
        paper, claim, evidence = fa3.records()
        self.assertEqual(paper.paper_id, "pap_flash_attention_3_neurips2024")
        self.assertEqual(
            claim.claim_id, "clm_fa3_varlen_external_cuseqlens_contract"
        )
        self.assertEqual(evidence.evidence_id, "evd_lit_fa3_varlen_api_contract")
        self.assertEqual(
            claim.scope["code_commit"],
            "c75d019dea9d910312974417bc28f190dfdda6d9",
        )
        self.assertEqual(evidence.paper_ids, (paper.paper_id,))
        self.assertEqual(evidence.claim_ids, (claim.claim_id,))

    def test_external_api_support_is_separate_from_local_transfer(self):
        _, _, evidence = fa3.records()
        self.assertEqual(
            evidence.assessment,
            {
                "relation": "supports",
                "strength": "strong",
                "limitations": [
                    "Strongly supports only the typed external cumulative-offset API claim.",
                    "Local packer-side boundary generation is explicitly not_tested/weak.",
                    "Local end-to-end H200 speed is explicitly not_tested/weak.",
                    "This record does not authorize implementation, launch, adoption, or SOTA.",
                ],
            },
        )
        local = evidence.facts["local_transfer_assessments"]
        for subject in (
            "ophis_packer_sidecar_generation",
            "ophis_end_to_end_h200_speed",
        ):
            self.assertEqual(local[subject]["relation"], "not_tested")
            self.assertEqual(local[subject]["strength"], "weak")

    def test_claim_requires_cumulative_offsets_and_maxima(self):
        _, claim, _ = fa3.records()
        self.assertEqual(
            claim.scope["sequence_metadata"],
            [
                "cu_seqlens_q",
                "cu_seqlens_k",
                "max_seqlen_q",
                "max_seqlen_k",
            ],
        )
        self.assertIn("saves them", claim.statement)
        self.assertIn("reuses them in backward", claim.statement)

    def test_capsule_manifest_and_inventory_exist(self):
        manifest_path = fa3.ARTIFACT_ROOT / "manifest.json"
        self.assertTrue(manifest_path.is_file())
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["license"]["path"], "LICENSE")
        self.assertEqual(
            manifest["license"]["sha256"],
            "8c9ccb96c065e706135b6cbad279b721da6156e51f3a5f27c6b3329af9416d73",
        )
        inventory = Path(fa3.INVENTORY_PATH).read_text(encoding="utf-8")
        for fragment in fa3.INVENTORY_REQUIRED_FRAGMENTS:
            self.assertIn(fragment, inventory)


if __name__ == "__main__":
    unittest.main()
