import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services import scenes

SCRIPT = (
    "Ahorrar energía en casa es fácil. "
    "Un primer consejo es instalar bombillas de bajo consumo, que duran mucho más. "
    "Mantener las cortinas abiertas ayuda. "
    "Desconecta los electrodomésticos que no uses."
)


def srt_for(lines):
    """lines: [(start, end, text)] -> file_to_subtitles() format."""
    def stamp(t):
        ms = int(round(t * 1000))
        return f"00:00:{ms // 1000:02d},{ms % 1000:03d}"
    return [(i + 1, f"{stamp(a)} --> {stamp(b)}", text) for i, (a, b, text) in enumerate(lines)]


class TestSceneTiming(unittest.TestCase):
    def test_split_sentences(self):
        self.assertEqual(len(scenes.split_sentences(SCRIPT)), 4)
        self.assertEqual(scenes.split_sentences("¿Hola? ¡Sí! Fin."), ["Hola", "Sí", "Fin"])

    def test_one_term_per_sentence_follows_subtitle_times(self):
        srt = srt_for([
            (0.0, 2.0, "Ahorrar energía en casa es fácil."),
            (2.0, 5.0, "Un primer consejo es instalar bombillas"),
            (5.0, 8.0, "de bajo consumo, que duran mucho más."),
            (8.0, 10.0, "Mantener las cortinas abiertas ayuda."),
            (10.0, 13.0, "Desconecta los electrodomésticos que no uses."),
        ])
        durations = scenes.scene_durations(SCRIPT, 4, [0, 1, 2, 3], srt, 13.0)
        self.assertEqual(len(durations), 4)
        self.assertAlmostEqual(sum(durations), 13.0, places=2)
        self.assertAlmostEqual(durations[0], 2.0, delta=0.15)
        self.assertAlmostEqual(durations[1], 6.0, delta=0.15)  # long sentence gets long scene
        self.assertAlmostEqual(durations[2], 2.0, delta=0.15)
        self.assertAlmostEqual(durations[3], 3.0, delta=0.15)

    def test_missing_clip_gives_time_to_previous_scene(self):
        srt = srt_for([(0, 2, "Ahorrar energía en casa es fácil."),
                       (2, 8, "Un primer consejo es instalar bombillas de bajo consumo, que duran mucho más."),
                       (8, 10, "Mantener las cortinas abiertas ayuda."),
                       (10, 13, "Desconecta los electrodomésticos que no uses.")])
        durations = scenes.scene_durations(SCRIPT, 4, [0, 1, 3], srt, 13.0)
        self.assertEqual(len(durations), 3)
        self.assertAlmostEqual(sum(durations), 13.0, places=2)
        self.assertAlmostEqual(durations[1], 8.0, delta=0.2)  # scene 2 absorbs missing scene 3

    def test_missing_first_clip_gives_time_to_next(self):
        durations = scenes.scene_durations(SCRIPT, 4, [1, 2, 3], [], 13.0)
        self.assertAlmostEqual(sum(durations), 13.0, places=2)

    def test_without_subtitles_uses_text_length(self):
        durations = scenes.scene_durations(SCRIPT, 4, [0, 1, 2, 3], [], 20.0)
        self.assertAlmostEqual(sum(durations), 20.0, places=2)
        self.assertGreater(durations[1], durations[0])

    def test_fewer_terms_than_sentences_groups_in_order(self):
        durations = scenes.scene_durations(SCRIPT, 2, [0, 1], [], 20.0)
        self.assertEqual(len(durations), 2)
        self.assertAlmostEqual(sum(durations), 20.0, places=2)

    def test_more_terms_than_sentences_splits_sentences(self):
        durations = scenes.scene_durations(SCRIPT, 8, list(range(8)), [], 20.0)
        self.assertEqual(len(durations), 8)
        self.assertAlmostEqual(sum(durations), 20.0, places=2)

    def test_short_scenes_get_minimum(self):
        script = "A. " + "Una frase muy larga " * 20 + "."
        durations = scenes.scene_durations(script, 2, [0, 1], [], 30.0)
        self.assertGreaterEqual(durations[0], scenes.MIN_SCENE_SECONDS - 1e-6)
        self.assertAlmostEqual(sum(durations), 30.0, places=2)

    def test_invalid_inputs_return_none(self):
        self.assertIsNone(scenes.scene_durations("", 3, [0], [], 10))
        self.assertIsNone(scenes.scene_durations(SCRIPT, 0, [0], [], 10))
        self.assertIsNone(scenes.scene_durations(SCRIPT, 3, [], [], 10))
        self.assertIsNone(scenes.scene_durations(SCRIPT, 3, [0], [], 0))

    def test_bad_srt_lines_are_ignored(self):
        bad = [(1, "garbage", "x"), (2, "00:00:01,000 --> nope", "y")]
        durations = scenes.scene_durations(SCRIPT, 4, [0, 1, 2, 3], bad, 12.0)
        self.assertAlmostEqual(sum(durations), 12.0, places=2)


if __name__ == "__main__":
    unittest.main()
