"""Replay successful HDF5 episodes into rubric-ready three-view MP4s.

Episodes must have been recorded in canonical target order: screwdriver,
wrench, hammer, pallet knife. Normal datasets contain recorded human actions.
Explicitly marked TEST-ONLY probe episodes are accepted solely for integration
testing, receive an on-frame warning, and cannot be written under ``datasets``.
"""

from __future__ import annotations

import argparse
import os
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Bin-PickPlace-Franka-Play-v0")
parser.add_argument("--dataset-file", type=Path, default=Path("datasets/bin_pickplace_demos.hdf5"))
parser.add_argument("--output-dir", type=Path, default=Path("datasets/videos"))
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--overwrite", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not args.enable_cameras:
    parser.error("video export requires --enable_cameras")

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import h5py
import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw

import bin_pickplace_tasks  # noqa: F401, E402
from isaaclab.utils.datasets import HDF5DatasetFileHandler  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from bin_pickplace_tasks.tasks.bin_pick_place import mdp  # noqa: E402
from bin_pickplace_tasks.tasks.bin_pick_place.mdp.events import (  # noqa: E402
    CANONICAL_INSTRUCTION,
    TOOL_DISPLAY_NAMES,
)


CAMERA_TERMS = (
    ("external_camera_1_rgb", "External camera 1"),
    ("external_camera_2_rgb", "External camera 2"),
    ("wrist_camera_rgb", "Wrist camera"),
)


def _rgb_image(tensor: torch.Tensor) -> Image.Image:
    return Image.fromarray(tensor[0].detach().cpu().numpy())


def _labeled_frame(
    policy: dict[str, torch.Tensor], instruction: str, success: bool, *, probe_only: bool = False
) -> Image.Image:
    """Create a 1280x1024, encoder-safe mosaic from synchronized policy images."""
    canvas = Image.new("RGB", (1280, 1024), color=(18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, 1279, 63), fill=(245, 245, 245))
    draw.text((18, 14), instruction, fill=(15, 15, 15))
    if probe_only:
        draw.text((18, 38), "AUTOMATED PIPELINE PROBE - NOT A HUMAN DEMONSTRATION", fill=(190, 0, 0))
    status = "metric: SUCCESS (detected)" if success else "metric: pending"
    draw.text((1010, 14), status, fill=((0, 125, 40) if success else (150, 75, 0)))

    placements = ((0, 64), (640, 64), (320, 544))
    for (term_name, label), placement in zip(CAMERA_TERMS, placements, strict=True):
        image = _rgb_image(policy[term_name])
        canvas.paste(image, placement)
        x, y = placement
        draw.rectangle((x, y, x + 170, y + 25), fill=(0, 0, 0))
        draw.text((x + 7, y + 5), label, fill=(255, 255, 255))
    return canvas


def main() -> None:
    env = None
    dataset = None
    try:
        if not args.dataset_file.is_file():
            raise FileNotFoundError(f"demonstration dataset not found: {args.dataset_file}")
        if args.fps <= 0:
            raise ValueError("--fps must be positive")

        dataset = HDF5DatasetFileHandler()
        dataset.open(str(args.dataset_file))
        episode_names = sorted(dataset.get_episode_names(), key=lambda name: int(name.rsplit("_", 1)[-1]))
        if len(episode_names) != len(TOOL_DISPLAY_NAMES):
            raise ValueError(
                f"expected exactly four successful episodes, found {len(episode_names)}: {episode_names}"
            )

        episode_specs = []
        with h5py.File(args.dataset_file, "r") as stream:
            probe_only = bool(stream["data"].attrs.get("probe_only", False))
            for episode_name in episode_names:
                episode_group = stream[f"data/{episode_name}"]
                required = ("target_tool_index", "target_tool_name", "language_instruction")
                missing = [name for name in required if name not in episode_group.attrs]
                if missing:
                    raise ValueError(
                        f"{episode_name} is missing Bin-PickPlace metadata {missing}; record with scripts/record_demos.py"
                    )
                target_index = int(episode_group.attrs["target_tool_index"])
                tool_name = str(episode_group.attrs["target_tool_name"])
                instruction = str(episode_group.attrs["language_instruction"])
                expected_instruction = CANONICAL_INSTRUCTION.format(tool_name=tool_name)
                if not 0 <= target_index < len(TOOL_DISPLAY_NAMES):
                    raise ValueError(f"{episode_name} has invalid target index {target_index}")
                if tool_name != TOOL_DISPLAY_NAMES[target_index] or instruction != expected_instruction:
                    raise ValueError(f"{episode_name} has inconsistent target/language metadata")
                episode_specs.append((target_index, episode_name, tool_name, instruction))
        if probe_only:
            print(
                "WARNING: exporting a TEST-ONLY automated pipeline probe, not human demonstrations",
                flush=True,
            )
            repository_datasets = (Path(__file__).resolve().parents[1] / "datasets").resolve()
            resolved_output_dir = args.output_dir.expanduser().resolve()
            if resolved_output_dir == repository_datasets or repository_datasets in resolved_output_dir.parents:
                raise ValueError(
                    f"TEST-ONLY probe output is forbidden under {repository_datasets}; use a /tmp directory"
                )
        episode_specs.sort()
        if [spec[0] for spec in episode_specs] != list(range(len(TOOL_DISPLAY_NAMES))):
            raise ValueError(f"dataset must contain each target exactly once: {episode_specs}")

        cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
        cfg.recorders = {}
        cfg.terminations = {}
        env = gym.make(args.task, cfg=cfg).unwrapped
        env.reset()
        args.output_dir.mkdir(parents=True, exist_ok=True)

        for target_index, episode_name, tool_name, instruction in episode_specs:
            output_path = args.output_dir / f"demo_{target_index}.mp4"
            partial_path = args.output_dir / f"demo_{target_index}.partial.mp4"
            if output_path.exists() and not args.overwrite:
                raise FileExistsError(f"refusing to overwrite {output_path}; pass --overwrite to replace it")
            if partial_path.exists():
                partial_path.unlink()

            episode = dataset.load_episode(episode_name, env.device)
            if episode is None or episode.get_initial_state() is None:
                raise ValueError(f"episode {episode_name} has no initial state")
            if episode.success is not None and not bool(episode.success):
                raise ValueError(f"episode {episode_name} is not marked successful")

            env.reset_to(episode.get_initial_state(), torch.tensor([0], device=env.device), is_relative=True)
            env.target_tool_index[:] = target_index
            env.success_streak.zero_()
            env.language_instructions[0] = instruction
            detected_success = False
            frame_count = 0

            writer = imageio.get_writer(
                partial_path,
                fps=args.fps,
                codec="libx264",
                pixelformat="yuv420p",
                macro_block_size=None,
            )
            try:
                while True:
                    action = episode.get_next_action()
                    if action is None:
                        break
                    batched_action = action if action.ndim == 2 else action.unsqueeze(0)
                    observations, _, _, _, _ = env.step(batched_action)
                    detected_success = detected_success or bool(mdp.target_placed(env)[0].item())
                    frame = _labeled_frame(
                        observations["policy"], instruction, detected_success, probe_only=probe_only
                    )
                    writer.append_data(np.asarray(frame))
                    frame_count += 1
            finally:
                writer.close()

            if not detected_success:
                partial_path.unlink(missing_ok=True)
                raise RuntimeError(f"{episode_name} replay ended without metric-detected success")
            partial_path.replace(output_path)
            print(f"exported {output_path} ({frame_count} frames, target={tool_name})", flush=True)

        artifact_kind = "TEST-ONLY probe videos" if probe_only else "demonstrations"
        print(f"Video export passed: four metric-complete H.264/yuv420p {artifact_kind}", flush=True)
    finally:
        if dataset is not None:
            dataset.close()
        if env is not None:
            env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        print(f"Video export failed: {error!r}", flush=True)
        traceback.print_exc()
        # Isaac Kit may terminate with status 0 from close() even after a
        # Python exception.  The environment has already closed in main().
        os._exit(1)
    simulation_app.close()
