"""Regression tests for Qwen relevance response parsing; no API key is needed."""
import unittest

from llm.skill_relevance import QwenSkillRelevanceScorer


CALLS = [
    {"id": "PickUp:red:ball", "name": "PickUp"},
    {"id": "UnlockDoor:yellow:door", "name": "UnlockDoor"},
]


class QwenResponseParsingTests(unittest.TestCase):
    def test_compact_numeric_json(self):
        result = QwenSkillRelevanceScorer._parse_score_response(
            '{"PickUp:red:ball":0.91,"UnlockDoor:yellow:door":0.74}', CALLS
        )
        self.assertEqual(result["PickUp:red:ball"]["relevance"], 0.91)
        self.assertEqual(result["UnlockDoor:yellow:door"]["relevance"], 0.74)

    def test_legacy_object_json_remains_supported(self):
        result = QwenSkillRelevanceScorer._parse_score_response(
            '{"PickUp:red:ball":{"relevance":0.9,"rationale":"next step"},'
            '"UnlockDoor:yellow:door":{"relevance":0.8}}',
            CALLS,
        )
        self.assertEqual(result["PickUp:red:ball"]["rationale"], "next step")
        self.assertEqual(result["UnlockDoor:yellow:door"]["relevance"], 0.8)

    def test_markdown_fenced_json_is_accepted(self):
        fence = chr(96) * 3
        response = fence + 'json\n{"PickUp:red:ball":0.9,"UnlockDoor:yellow:door":0.8}\n' + fence
        result = QwenSkillRelevanceScorer._parse_score_response(response, CALLS)
        self.assertEqual(len(result), 2)

    def test_truncated_json_is_rejected(self):
        with self.assertRaises(ValueError):
            QwenSkillRelevanceScorer._parse_score_response(
                '{"PickUp:red:ball":0.9,"UnlockDoor:yellow:door":',
                CALLS,
            )

    def test_missing_or_unexpected_candidate_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "candidate IDs mismatch"):
            QwenSkillRelevanceScorer._parse_score_response(
                '{"PickUp:red:ball":0.9}', CALLS
            )
        with self.assertRaisesRegex(ValueError, "candidate IDs mismatch"):
            QwenSkillRelevanceScorer._parse_score_response(
                '{"PickUp:red:ball":0.9,"UnlockDoor:yellow:door":0.8,"Other":0.1}',
                CALLS,
            )

    def test_out_of_range_nonfinite_and_boolean_scores_are_rejected(self):
        for bad_score in ("1.1", "-0.1", "NaN", "true"):
            with self.subTest(score=bad_score), self.assertRaises(ValueError):
                QwenSkillRelevanceScorer._parse_score_response(
                    '{"PickUp:red:ball":' + bad_score
                    + ',"UnlockDoor:yellow:door":0.8}',
                    CALLS,
                )


if __name__ == "__main__":
    unittest.main()
