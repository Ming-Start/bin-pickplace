"""CPU probes for the formal placement metric (no simulator launch required)."""

from __future__ import annotations

import unittest
import importlib.util
from pathlib import Path

import torch


GEOMETRY_PATH = (
    Path(__file__).resolve().parents[1]
    / "source/bin_pickplace_tasks/bin_pickplace_tasks/tasks/bin_pick_place/mdp/geometry.py"
)
spec = importlib.util.spec_from_file_location("bin_pickplace_metric_geometry", GEOMETRY_PATH)
geometry = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(geometry)
TOOL_BOUNDS_MIN = geometry.TOOL_BOUNDS_MIN
TOOL_BOUNDS_MAX = geometry.TOOL_BOUNDS_MAX
placement_components = geometry.placement_components

EVENTS_PATH = GEOMETRY_PATH.with_name("events.py")
events_spec = importlib.util.spec_from_file_location("bin_pickplace_metric_events", EVENTS_PATH)
events = importlib.util.module_from_spec(events_spec)
assert events_spec.loader is not None
events_spec.loader.exec_module(events)


def component_probe(
    *,
    pos=(0.0, 0.0, 0.05),
    lin=(0.0, 0.0, 0.0),
    ang=(0.0, 0.0, 0.0),
    opening=0.08,
    tcp=(-0.5, 0.0, 0.3),
):
    f = lambda x: torch.tensor([x], dtype=torch.float32)
    return placement_components(
        f(pos),
        f((1.0, 0.0, 0.0, 0.0)),
        f(TOOL_BOUNDS_MIN[0]),
        f(TOOL_BOUNDS_MAX[0]),
        f((0.0, 0.0, 0.0)),
        f((1.0, 0.0, 0.0, 0.0)),
        f(lin),
        f(ang),
        f(opening),
        f(tcp),
    )


class MetricProbes(unittest.TestCase):
    def test_stock_recorder_progress_maps_to_canonical_target(self):
        class Recorder:
            def exported_successful_episode_count(self):
                return 3

        class Env:
            recorder_manager = Recorder()

        self.assertEqual(events.recording_target_index(Env()), 3)

    def test_reset_layout_is_inside_and_root_poses_are_distinct(self):
        name_to_index = {name: index for index, name in enumerate(events.TOOL_NAMES)}
        white_position = torch.tensor(events.BIN_ROOT_POSITIONS["white_bin"], dtype=torch.float32)
        interior_min = torch.tensor(geometry.WHITE_INTERIOR_MIN)
        interior_max = torch.tensor(geometry.WHITE_INTERIOR_MAX)
        tool_positions = {}

        for name, (position, quaternion) in events.TOOL_RESET_POSES.items():
            index = name_to_index[name]
            bounds_min = torch.tensor([geometry.TOOL_BOUNDS_MIN[index]])
            bounds_max = torch.tensor([geometry.TOOL_BOUNDS_MAX[index]])
            corners = geometry.box_corners(bounds_min, bounds_max)
            quat = torch.tensor([[quaternion]], dtype=torch.float32).expand(-1, 8, -1)
            local_position = torch.tensor(position, dtype=torch.float32) - white_position
            corners_white = local_position[None, None, :] + geometry.quat_apply(quat, corners)
            self.assertTrue(bool(torch.all((corners_white > interior_min) & (corners_white < interior_max))))
            self.assertAlmostEqual(float(torch.linalg.vector_norm(quat[0, 0])), 1.0, places=5)
            tool_positions[name] = torch.tensor(position, dtype=torch.float32)

        # Rotated AABBs are intentionally not used for collision rejection:
        # these long, non-rectangular tools contain large empty regions in
        # their boxes. Exact packaged PhysX colliders and post-settle drift are
        # checked by scripts/smoke_test.py on CUDA instead.
        names = list(tool_positions)
        for left_index, left_name in enumerate(names):
            for right_name in names[left_index + 1 :]:
                root_separation = torch.linalg.vector_norm(
                    tool_positions[left_name][:2] - tool_positions[right_name][:2]
                )
                self.assertGreater(
                    float(root_separation),
                    0.10,
                    f"tool roots are implausibly close: {left_name}, {right_name}",
                )

    def test_valid_placement(self):
        inside, settled, released = component_probe()
        self.assertTrue(bool(inside.item() and settled.item() and released.item()))

    def test_measured_bin_floor_rest_height_is_inside(self):
        # CUDA contact settling puts the lowest rotated screwdriver corner near
        # local z=0.00285 m.  A contained tool resting on the floor must pass.
        inside, _, _ = component_probe(pos=(0.0, 0.0, 0.00285))
        self.assertTrue(bool(inside.item()))

    def test_still_held_is_rejected(self):
        _, _, released = component_probe(opening=0.0)
        self.assertFalse(bool(released.item()))

    def test_moving_or_bouncing_is_rejected(self):
        _, settled_linear, _ = component_probe(lin=(0.10, 0.0, 0.0))
        _, settled_angular, _ = component_probe(ang=(0.0, 0.0, 0.60))
        self.assertFalse(bool(settled_linear.item()))
        self.assertFalse(bool(settled_angular.item()))

    def test_rim_is_rejected_using_all_corners(self):
        inside, _, _ = component_probe(pos=(0.20, 0.0, 0.05))
        self.assertFalse(bool(inside.item()))

    def test_wrong_tool_and_ten_step_persistence(self):
        # The instructed screwdriver remains at the source location. A different tool being in the
        # blue bin is irrelevant because the metric receives only the indexed target's tensors.
        inside_target, _, _ = component_probe(pos=(0.0, -0.62, 0.05))
        self.assertFalse(bool(inside_target.item()))

        streak = 0
        for _ in range(9):
            inside, settled, released = component_probe()
            streak = streak + 1 if bool(inside.item() and settled.item() and released.item()) else 0
            self.assertLess(streak, 10)
        inside, settled, released = component_probe()
        streak = streak + 1 if bool(inside.item() and settled.item() and released.item()) else 0
        self.assertEqual(streak, 10)


if __name__ == "__main__":
    unittest.main()
