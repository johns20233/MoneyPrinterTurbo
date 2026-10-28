import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services import scene_finish as sf

LINES = [(0.0, 2.0, "Ahorra energía en casa"), (2.5, 4.0, "Primero"), (5.0, 8.0, "abre las cortinas durante el día")]


class TestSceneFinish(unittest.TestCase):
    def test_word_times_cover_each_line(self):
        words = sf.word_times(LINES)
        self.assertEqual([w[2] for w in words][:4], ["Ahorra", "energía", "en", "casa"])
        self.assertAlmostEqual(words[0][0], 0.0)
        self.assertAlmostEqual(words[3][1], 2.0)

    def test_chunks_respect_limits_and_lines(self):
        groups = sf.chunks(sf.word_times(LINES))
        for g in groups:
            self.assertLessEqual(len(g), sf.MAX_WORDS)
            self.assertEqual(len({w[3] for w in g}), 1)

    def test_states_cover_timeline_without_overlap(self):
        timeline, groups = sf.states(sf.word_times(LINES), 9.0)
        self.assertAlmostEqual(timeline[0][0], 0.0)
        self.assertAlmostEqual(timeline[-1][1], 9.0)
        for a, b in zip(timeline, timeline[1:]):
            self.assertAlmostEqual(a[1], b[0], places=6)
        # long pause (4.0 -> 5.0) shows no subtitle
        self.assertTrue(any(s[2] is None and s[0] >= 3.9 and s[1] <= 5.1 for s in timeline))

    def test_spanish_text_supported_other_scripts_fall_back(self):
        self.assertTrue(sf.supports_text("¿Ahorrás energía? Sí, ¡muchísimo! 80% €"))
        self.assertFalse(sf.supports_text("节约能源"))
        self.assertFalse(sf.supports_text("Экономия энергии"))


if __name__ == "__main__":
    unittest.main()
