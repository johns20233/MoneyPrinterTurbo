import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.config import config
from app.models.schema import MaterialInfo
from app.services import relevance


def _item(n, thumb=True):
    item = MaterialInfo()
    item.provider = "pexels"
    item.url = f"https://videos.example/{n}.mp4"
    item.duration = 10
    item.source_info = {"provider": "pexels", "asset_id": str(n)}
    if thumb:
        item.source_info["thumbnail"] = f"https://images.example/{n}.jpg"
    return item


class TestVisualRelevance(unittest.TestCase):
    def setUp(self):
        self.original = dict(config.app)
        config.app["visual_relevance_check"] = True
        config.app["openai_api_key"] = "test-key"

    def tearDown(self):
        config.app.clear()
        config.app.update(self.original)

    def test_disabled_returns_items_unchanged_without_calling_vision(self):
        config.app["visual_relevance_check"] = False
        items = [_item(1), _item(2)]
        with patch.object(relevance, "_ask_vision") as ask:
            self.assertEqual(relevance.rank_by_relevance("light bulb", items), items)
        ask.assert_not_called()

    def test_keeps_only_approved_clips_best_first(self):
        items = [_item(1), _item(2), _item(3)]
        with patch.object(relevance, "_ask_vision", return_value=[2, 9, 7]):
            result = relevance.rank_by_relevance("electricity meter", items)
        self.assertEqual([i.source_info["asset_id"] for i in result], ["2", "3"])
        self.assertEqual(result[0].source_info["relevance_score"], 9)

    def test_all_rejected_returns_empty_list_never_unrelated_clips(self):
        items = [_item(1), _item(2)]
        with patch.object(relevance, "_ask_vision", return_value=[1, 3]):
            self.assertEqual(relevance.rank_by_relevance("home insulation", items), [])

    def test_reviews_at_most_max_candidates_in_one_request(self):
        items = [_item(n) for n in range(12)]
        with patch.object(relevance, "_ask_vision", return_value=[8] * relevance.MAX_CANDIDATES) as ask:
            result = relevance.rank_by_relevance("study notes", items)
        ask.assert_called_once()
        self.assertEqual(len(ask.call_args.args[1]), relevance.MAX_CANDIDATES)
        self.assertEqual(len(result), relevance.MAX_CANDIDATES)

    def test_fail_open_on_error_or_invalid_answer(self):
        items = [_item(1), _item(2)]
        with patch.object(relevance, "_ask_vision", side_effect=TimeoutError("slow")):
            self.assertEqual(relevance.rank_by_relevance("light bulb", items), items)
        with patch.object(relevance, "_ask_vision", return_value=None):
            self.assertEqual(relevance.rank_by_relevance("light bulb", items), items)

    def test_items_without_preview_are_returned_unchanged(self):
        items = [_item(1, thumb=False), _item(2, thumb=False)]
        with patch.object(relevance, "_ask_vision") as ask:
            self.assertEqual(relevance.rank_by_relevance("light bulb", items), items)
        ask.assert_not_called()

    def test_parse_scores_validates_shape(self):
        self.assertEqual(relevance._parse_scores('{"scores":[3, 12, -1]}', 3), [3, 10, 0])
        self.assertIsNone(relevance._parse_scores('{"scores":[3]}', 2))
        self.assertIsNone(relevance._parse_scores("not json", 1))
        self.assertIsNone(relevance._parse_scores('{"scores":[true, 5]}', 2))

    def test_no_api_key_means_no_vision_call(self):
        config.app["openai_api_key"] = ""
        self.assertIsNone(relevance._ask_vision("light bulb", ["https://images.example/1.jpg"]))


if __name__ == "__main__":
    unittest.main()
