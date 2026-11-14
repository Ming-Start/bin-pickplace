"""Controlled Bin-PickPlace compatibility profile for the stock Isaac Lab replay tool."""

from __future__ import annotations

from pathlib import Path

import h5py
import torch


APP_LAUNCHER_ANCHOR = "simulation_app = app_launcher.app"
RESET_TO_ANCHOR = (
    "env.reset_to(initial_state, torch.tensor([env_id], device=env.device), is_relative=True)"
)
POST_APP_INSTALLATION = (
    "from replay_profile import install_state_validation_compatibility\n"
    "install_state_validation_compatibility()"
)
POST_RESET_RESTORE = (
    "from replay_profile import restore_episode_target\n"
    "                            restore_episode_target(\n"
    "                                env, env_id, args_cli.dataset_file, "
    "episode_names[next_episode_index], next_episode_index\n"
    "                            )"
)

TOOL_DISPLAY_NAMES = ("screwdriver", "wrench", "hammer", "pallet knife")
CANONICAL_INSTRUCTION = "Pick the {tool_name} from the white bin and place it in the blue bin."


def inject_profile(source: str) -> str:
    """Inject state compatibility and target restoration at verified anchors."""
    if source.count(APP_LAUNCHER_ANCHOR) != 1:
        raise RuntimeError(
            "Isaac Lab replay_demos.py changed: expected exactly one AppLauncher anchor; "
            "refusing an unsafe patch"
        )
    if source.count(RESET_TO_ANCHOR) != 1:
        raise RuntimeError(
            "Isaac Lab replay_demos.py changed: expected exactly one reset_to anchor; "
            "refusing to restore targets at an unverified location"
        )
    source = source.replace(
        APP_LAUNCHER_ANCHOR,
        APP_LAUNCHER_ANCHOR + "\n\n" + POST_APP_INSTALLATION,
        1,
    )
    return source.replace(
        RESET_TO_ANCHOR,
        RESET_TO_ANCHOR + "\n                            " + POST_RESET_RESTORE,
        1,
    )


def _remove_stock_synthetic_batch(value):
    """Remove only the leading singleton dimension added by stock EpisodeData.get_state."""
    if isinstance(value, dict):
        return {key: _remove_stock_synthetic_batch(item) for key, item in value.items()}
    if isinstance(value, torch.Tensor) and value.ndim >= 2 and value.shape[0] == 1:
        return value.squeeze(0)
    return value


def install_state_validation_compatibility() -> None:
    """Make stock ``--validate_states`` compare equivalent tensors and shapes."""
    from isaaclab.utils.datasets import EpisodeData

    if getattr(EpisodeData, "_bin_pickplace_state_shape_compatibility", False):
        return

    original_get_state = EpisodeData.get_state

    def get_state_without_synthetic_batch(self, state_index):
        return _remove_stock_synthetic_batch(original_get_state(self, state_index))

    EpisodeData.get_state = get_state_without_synthetic_batch
    EpisodeData._bin_pickplace_state_shape_compatibility = True
    print(
        "[Bin-PickPlace] replay state validation compatibility enabled: "
        "only the synthetic singleton batch dimension is removed.",
        flush=True,
    )


def restore_episode_target(env, env_id: int, dataset_file: str, episode_name: str, fallback_index: int) -> None:
    """Restore the target and instruction from episode metadata after a stock reset."""
    dataset_path = Path(dataset_file).expanduser().resolve()
    with h5py.File(dataset_path, "r") as stream:
        episode = stream[f"data/{episode_name}"]
        target_index = int(episode.attrs.get("target_tool_index", fallback_index))
        target_name = str(episode.attrs.get("target_tool_name", TOOL_DISPLAY_NAMES[target_index % 4]))
        instruction = str(
            episode.attrs.get("language_instruction", CANONICAL_INSTRUCTION.format(tool_name=target_name))
        )

    if not 0 <= target_index < len(TOOL_DISPLAY_NAMES):
        raise ValueError(f"{episode_name} has out-of-range target_tool_index={target_index}")
    env.target_tool_index[env_id] = target_index
    env.success_streak[env_id] = 0
    env.language_instructions[env_id] = instruction
    print(
        f"[Bin-PickPlace][replay env_{env_id}] {episode_name}: {instruction}",
        flush=True,
    )
