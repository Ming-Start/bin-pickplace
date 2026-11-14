"""CPU-only regression tests for the demonstration recording filter."""

from __future__ import annotations

import unittest

import torch

from scripts.recording_profile import (
    APP_LAUNCHER_ANCHOR,
    POST_APP_INSTALLATION,
    filter_policy_observations,
    inject_profile_after_app_launcher,
)


class RecordingProfileTests(unittest.TestCase):
    def test_stock_source_hook_is_inserted_immediately_after_app_launcher(self):
        source = f"before\n{APP_LAUNCHER_ANCHOR}\nafter\n"
        patched = inject_profile_after_app_launcher(source)
        self.assertIn(
            f"{APP_LAUNCHER_ANCHOR}\n\n{POST_APP_INSTALLATION}\nafter",
            patched,
        )

    def test_stock_source_change_fails_closed(self):
        with self.assertRaisesRegex(RuntimeError, "AppLauncher anchor"):
            inject_profile_after_app_launcher("no matching anchor")

    def test_camera_terms_are_excluded_but_low_dimensional_contract_is_retained(self):
        policy = {
            "joint_positions": torch.zeros((1, 9)),
            "tool_poses": torch.zeros((1, 28)),
            "previous_action": torch.zeros((1, 7)),
            "external_camera_1_rgb": torch.zeros((1, 480, 640, 3), dtype=torch.uint8),
            "external_camera_2_rgb": torch.zeros((1, 480, 640, 3), dtype=torch.uint8),
            "wrist_camera_rgb": torch.zeros((1, 480, 640, 3), dtype=torch.uint8),
        }
        filtered = filter_policy_observations(policy)
        self.assertEqual(
            set(filtered), {"joint_positions", "tool_poses", "previous_action"}
        )

    def test_future_high_dimensional_image_terms_are_excluded_defensively(self):
        policy = {
            "compact_feature": torch.zeros((1, 32)),
            "future_camera": torch.zeros((1, 64, 64, 4)),
            "front_depth": torch.zeros((1, 64, 64, 1)),
        }
        self.assertEqual(set(filter_policy_observations(policy)), {"compact_feature"})


if __name__ == "__main__":
    unittest.main()
