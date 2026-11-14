"""CPU regression tests for the Bin-PickPlace stock replay compatibility profile."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import h5py
import torch

from scripts.replay_profile import (
    APP_LAUNCHER_ANCHOR,
    POST_APP_INSTALLATION,
    RESET_TO_ANCHOR,
    _remove_stock_synthetic_batch,
    inject_profile,
    restore_episode_target,
)


class ReplayProfileTests(unittest.TestCase):
    def test_injection_requires_both_stock_source_anchors(self):
        source = f"{APP_LAUNCHER_ANCHOR}\n{RESET_TO_ANCHOR}\n"
        patched = inject_profile(source)
        self.assertIn(POST_APP_INSTALLATION, patched)
        self.assertIn("restore_episode_target", patched)
        with self.assertRaisesRegex(RuntimeError, "AppLauncher"):
            inject_profile(RESET_TO_ANCHOR)
        with self.assertRaisesRegex(RuntimeError, "reset_to"):
            inject_profile(APP_LAUNCHER_ANCHOR)

    def test_only_synthetic_leading_batch_is_removed(self):
        synthetic_state = {
            "root_pose": torch.arange(7, dtype=torch.float32).reshape(1, 7),
            "nested_pose": torch.arange(7, dtype=torch.float32).reshape(1, 1, 7),
        }
        state = _remove_stock_synthetic_batch(synthetic_state)
        self.assertEqual(tuple(state["root_pose"].shape), (7,))
        self.assertEqual(tuple(state["nested_pose"].shape), (1, 7))

    def test_episode_metadata_restores_target_after_stock_reset(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "demo.hdf5"
            with h5py.File(path, "w") as stream:
                episode = stream.create_group("data/demo_0")
                episode.attrs["target_tool_index"] = 0
                episode.attrs["target_tool_name"] = "screwdriver"
                episode.attrs["language_instruction"] = (
                    "Pick the screwdriver from the white bin and place it in the blue bin."
                )
            env = SimpleNamespace(
                target_tool_index=torch.tensor([1]),
                success_streak=torch.tensor([9]),
                language_instructions=["wrong"],
            )
            restore_episode_target(env, 0, str(path), "demo_0", 3)
            self.assertEqual(env.target_tool_index.tolist(), [0])
            self.assertEqual(env.success_streak.tolist(), [0])
            self.assertIn("screwdriver", env.language_instructions[0])


if __name__ == "__main__":
    unittest.main()
