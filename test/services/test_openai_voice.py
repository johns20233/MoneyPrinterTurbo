import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services import openai_voice as ov


def lines_of(text):
    return [x.strip() for x in re.split(r"[,.;:!?\n]", text) if x.strip()]


class TestOpenAIVoice(unittest.TestCase):
    def test_split_keeps_sentence_punctuation(self):
        chunks = ov.split_chunks("Hola mundo. ¿Qué tal? Bien: gracias.\nFin")
        self.assertEqual(chunks, ["Hola mundo.", "¿Qué tal?", "Bien:", "gracias.", "Fin"])

    def test_decimals_are_not_split(self):
        self.assertEqual(ov.split_chunks("Cuesta 2.5 euros. Listo."), ["Cuesta 2.5 euros.", "Listo."])

    def test_assign_lines_times_follow_sentences_and_gaps(self):
        items = ov.assign_lines(["Uno, dos.", "Tres."], [2.0, 1.0], 0.2, lines_of)
        self.assertEqual([i[0] for i in items], ["Uno", "dos", "Tres"])
        self.assertAlmostEqual(items[0][1], 0.0)
        self.assertAlmostEqual(items[1][2], 2.0)
        self.assertAlmostEqual(items[2][1], 2.2)
        self.assertAlmostEqual(items[2][2], 3.2)

    def test_instructions_name_the_accent(self):
        self.assertIn("Spain", ov.instructions("es-ES"))
        self.assertIn("xx-YY", ov.instructions("xx-YY"))


if __name__ == "__main__":
    unittest.main()
