import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.models.schema import VideoAspect
from app.services import ai_images as ai

LONG = " ".join(["palabra"] * 20)


class TestAIImages(unittest.TestCase):
    def test_empty_scenes_first_then_long_single_scenes(self):
        picked = ai.pick_scenes(5, [1, 0, 2, 1, 0], ["corta", "", LONG, LONG + " extra", "x"], 6)
        self.assertEqual(picked, [1, 4, 3])

    def test_short_single_scenes_are_not_picked(self):
        self.assertEqual(ai.pick_scenes(2, [1, 1], ["hola que tal", "bien"], 6), [])

    def test_cap_is_respected(self):
        self.assertEqual(ai.pick_scenes(10, [0] * 10, None, 6), [0, 1, 2, 3, 4, 5])
        self.assertEqual(ai.pick_scenes(3, [0, 0, 0], None, 0), [])

    def test_prompt_forbids_text_and_real_people(self):
        prompt = ai.build_prompt("smart meter on wall", 'Revisa el "contador" de luz.', VideoAspect.portrait)
        self.assertIn("vertical", prompt)
        self.assertIn("smart meter on wall", prompt)
        self.assertIn("no text", prompt.lower())
        self.assertIn("No recognisable real", prompt)
        self.assertNotIn('"contador"', prompt)

    def test_disabled_by_default_makes_no_request(self):
        with mock.patch.object(ai, "generate_clip") as gen, \
             mock.patch.dict(ai.config.app, {"ai_scene_images": False, "openai_api_key": "k"}):
            self.assertEqual(ai.fill_scenes(["a"], [0], None, VideoAspect.portrait, "/tmp"), [])
            gen.assert_not_called()

    def test_enabled_fills_only_chosen_scenes_and_skips_failures(self):
        results = {0: "/tmp/a.mp4", 2: None}
        with mock.patch.object(ai, "generate_clip", side_effect=lambda t, n, a, d: results[["a", "b", "c"].index(t)]), \
             mock.patch.dict(ai.config.app, {"ai_scene_images": True, "openai_api_key": "k", "ai_scene_images_max": 6}):
            made = ai.fill_scenes(["a", "b", "c"], [0, 3, 0], None, VideoAspect.portrait, "/tmp")
        self.assertEqual(made, [(0, "/tmp/a.mp4")])

    def test_no_key_means_disabled(self):
        with mock.patch.dict(ai.config.app, {"ai_scene_images": True, "openai_api_key": ""}):
            self.assertFalse(ai.enabled())


if __name__ == "__main__":
    unittest.main()
