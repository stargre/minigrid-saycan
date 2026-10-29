"""Regression tests for Qwen relevance response parsing; no API key is needed."""
import unittest
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import Mock

from llm.skill_relevance import QwenSkillRelevanceScorer


CALLS = [
    {"id": "PickUp:red:ball", "name": "PickUp"},
    {"id": "UnlockDoor:yellow:door", "name": "UnlockDoor"},
]


def fake_response(content):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


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

    def test_malformed_response_is_retried_once_then_cached(self):
        scorer = QwenSkillRelevanceScorer.__new__(QwenSkillRelevanceScorer)
        scorer.model = "mock-qwen"
        scorer.client = Mock()
        scorer.client.chat.completions.create.side_effect = [
            fake_response('{"PickUp:red:ball":0.9,"UnlockDoor:yellow:door":'),
            fake_response('{"PickUp:red:ball":0.9,"UnlockDoor:yellow:door":0.8}'),
        ]
        scorer.last_latency_s = 0.0
        scorer.request_count = 0
        scorer.api_latencies_s = []
        scorer.parse_retry_count = 0
        scorer._cache = OrderedDict()
        scorer.cache_size = 8

        result = scorer.score("Pick up the red ball and unlock the door", CALLS, {})

        self.assertEqual(result["PickUp:red:ball"]["relevance"], 0.9)
        self.assertEqual(scorer.request_count, 2)
        self.assertEqual(scorer.parse_retry_count, 1)
        self.assertEqual(len(scorer.api_latencies_s), 2)
        self.assertEqual(scorer.client.chat.completions.create.call_count, 2)

        cached = scorer.score("Pick up the red ball and unlock the door", CALLS, {})
        self.assertEqual(cached, result)
        self.assertEqual(scorer.client.chat.completions.create.call_count, 2)


if __name__ == "__main__":
    unittest.main()
