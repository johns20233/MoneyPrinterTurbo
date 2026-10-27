import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services import scene_render as sr

LENGTHS = {f"c{i}.mp4": 8.0 for i in range(20)}
LENGTHS["short.mp4"] = 2.0


class TestScenePlan(unittest.TestCase):
    def test_frames_are_exact_and_scenes_start_on_time(self):
        durations = [4.31, 1.2, 7.93, 5.11]
        groups = [["c1.mp4", "c2.mp4"], ["c3.mp4"], ["c4.mp4", "c5.mp4", "c6.mp4"], ["c7.mp4", "c8.mp4"]]
        plan = sr.plan(durations, groups, LENGTHS)
        self.assertEqual(sum(s["frames"] for s in plan), round(sum(durations) * sr.FPS))
        for scene in plan:
            self.assertEqual(sum(shot["frames"] for shot in scene["shots"]), scene["rendered"])
            extra = round(sr.TRANSITION_SECONDS * sr.FPS) if scene["transition"] else 0
            self.assertEqual(scene["rendered"], scene["frames"] + extra)

    def test_never_reuses_a_clip_and_moves_change(self):
        groups = [[f"c{3 * i + k}.mp4" for k in range(3)] for i in range(5)]
        plan = sr.plan([6.0] * 5, groups, LENGTHS)
        shots = [shot for scene in plan for shot in scene["shots"]]
        self.assertEqual(len({s["path"] for s in shots}), len(shots))
        for a, b in zip(shots, shots[1:]):
            self.assertNotEqual(a["motion"], b["motion"])

    def test_one_clip_scene_is_one_shot(self):
        plan = sr.plan([9.0], [["c1.mp4"]], LENGTHS)
        self.assertEqual(len(plan[0]["shots"]), 1)
        self.assertIsNone(plan[0]["transition"])

    def test_short_clip_is_slowed_but_not_below_minimum(self):
        plan = sr.plan([6.0], [["short.mp4"]], LENGTHS)
        self.assertAlmostEqual(plan[0]["shots"][0]["speed"], sr.MIN_SPEED)

    def test_no_transition_around_very_short_scenes(self):
        plan = sr.plan([4.0, 1.0, 4.0], [["c1.mp4"], ["c2.mp4"], ["c3.mp4"]], LENGTHS)
        self.assertIsNone(plan[0]["transition"])
        self.assertIsNone(plan[1]["transition"])

    def test_transitions_vary(self):
        plan = sr.plan([4.0] * 4, [[f"c{i}.mp4"] for i in range(4)], LENGTHS)
        names = [s["transition"] for s in plan[:-1]]
        self.assertEqual(len(set(names)), 3)
        self.assertIsNone(plan[-1]["transition"])

    def test_filtergraph_uses_xfade_offsets_at_scene_starts(self):
        plan = sr.plan([4.0, 5.0, 3.0], [["c1.mp4"], ["c2.mp4", "c3.mp4"], ["c4.mp4"]], LENGTHS)
        graph, label = sr.compose_filter(plan)
        self.assertIn("offset=4.0000", graph)
        self.assertIn("offset=9.0000", graph)
        self.assertEqual(label, "[j2]")
        hard, _ = sr.compose_filter(plan, transitions=False)
        self.assertNotIn("xfade", hard)

    def test_invalid_plans_raise(self):
        with self.assertRaises(ValueError):
            sr.plan([3.0], [[]], LENGTHS)
        with self.assertRaises(ValueError):
            sr.plan([3.0, 2.0], [["c1.mp4"]], LENGTHS)


if __name__ == "__main__":
    unittest.main()
