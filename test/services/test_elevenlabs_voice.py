import base64
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services import elevenlabs_voice as ev


def timings(text, per_char=0.1):
    chars = list(text)
    starts = [i * per_char for i in range(len(chars))]
    ends = [s + per_char for s in starts]
    return chars, starts, ends


class TestElevenLabsVoice(unittest.TestCase):
    def test_lines_get_exact_times_and_stay_until_next(self):
        text = "Apaga la luz, ahorra. Cambia bombillas."
        chars, starts, ends = timings(text)
        items = ev.align_lines(["Apaga la luz", "ahorra", "Cambia bombillas"], chars, starts, ends)
        self.assertEqual([i[0] for i in items], ["Apaga la luz", "ahorra", "Cambia bombillas"])
        self.assertAlmostEqual(items[0][1], 0.0)
        self.assertAlmostEqual(items[1][1], text.index("ahorra") * 0.1)
        self.assertAlmostEqual(items[0][2], items[1][1])           # no gap between lines
        self.assertAlmostEqual(items[2][2], (text.index("bombillas") + 9) * 0.1)

    def test_matching_ignores_case_accents_spacing_and_punctuation(self):
        text = "¿Sabías  que...  el LED ahorra?"
        chars, starts, ends = timings(text)
        items = ev.align_lines(["Sabías que", "el led ahorra"], chars, starts, ends)
        self.assertIsNotNone(items)
        self.assertEqual(len(items), 2)

    def test_missing_line_returns_none(self):
        chars, starts, ends = timings("Hola mundo")
        self.assertIsNone(ev.align_lines(["Adiós"], chars, starts, ends))
        self.assertIsNone(ev.align_lines(["Hola"], chars, starts[:-1], ends))

    def test_disabled_without_key_or_voice(self):
        with mock.patch.dict(ev.config.elevenlabs, {"api_key": "", "voice_male": "v"}):
            self.assertFalse(ev.enabled_for("Male"))
        with mock.patch.dict(ev.config.elevenlabs, {"api_key": "k", "voice_male": "v", "voice_female": ""}):
            self.assertTrue(ev.enabled_for("Male"))
            self.assertFalse(ev.enabled_for("Female"))

    def test_synthesize_writes_audio_and_returns_lines(self):
        text = "Apaga la luz. Ahorra cada mes."
        chars, starts, ends = timings(text)
        body = {
            "audio_base64": base64.b64encode(b"x" * 2000).decode(),
            "alignment": {"characters": chars, "character_start_times_seconds": starts,
                          "character_end_times_seconds": ends},
        }
        response = mock.Mock(status_code=200, json=lambda: body)
        import tempfile, os
        out = os.path.join(tempfile.mkdtemp(), "voice.mp3")
        with mock.patch.object(ev.requests, "post", return_value=response) as post, \
             mock.patch.dict(ev.config.elevenlabs, {"api_key": "k", "voice_male": "david"}):
            items = ev.synthesize(text, "Male", 1.0, out, lambda t: ["Apaga la luz", "Ahorra cada mes"])
        self.assertEqual([i[0] for i in items], ["Apaga la luz", "Ahorra cada mes"])
        self.assertTrue(os.path.getsize(out) == 2000)
        self.assertIn("/david/with-timestamps", post.call_args[0][0])
        self.assertEqual(post.call_args[1]["json"]["voice_settings"]["speed"], 0.9)

    def test_http_error_returns_none(self):
        response = mock.Mock(status_code=401, text="bad key")
        with mock.patch.object(ev.requests, "post", return_value=response), \
             mock.patch.dict(ev.config.elevenlabs, {"api_key": "k", "voice_male": "david"}):
            self.assertIsNone(ev.synthesize("Hola.", "Male", 1.0, "/tmp/x.mp3", lambda t: ["Hola"]))


if __name__ == "__main__":
    unittest.main()
