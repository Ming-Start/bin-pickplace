"""Run Isaac Lab's stock recorder with the Bin-PickPlace low-memory RGB filtering."""

import os
import sys
from pathlib import Path

import h5py

from _stock_tool_launcher import launch_stock_tool


TOOL_DISPLAY_NAMES = ("screwdriver", "wrench", "hammer", "pallet knife")
CANONICAL_INSTRUCTION = "Pick the {tool_name} from the white bin and place it in the blue bin."


def _dataset_path(argv: list[str]) -> Path:
    for index, argument in enumerate(argv):
        if argument == "--dataset_file" and index + 1 < len(argv):
            return Path(argv[index + 1])
        if argument.startswith("--dataset_file="):
            return Path(argument.split("=", 1)[1])
    return Path("datasets/dataset.hdf5")


def annotate_dataset(path: Path) -> int:
    """Persist target/language metadata omitted by Isaac Lab's stock recorder."""
    if not path.is_file():
        return 0
    with h5py.File(path, "r+") as stream:
        data = stream.get("data")
        if data is None:
            raise ValueError(f"HDF5 file has no /data group: {path}")
        episode_names = sorted(data.keys(), key=lambda name: int(name.rsplit("_", 1)[-1]))
        for ordinal, episode_name in enumerate(episode_names):
            target_index = ordinal % len(TOOL_DISPLAY_NAMES)
            tool_name = TOOL_DISPLAY_NAMES[target_index]
            episode = data[episode_name]
            episode.attrs["target_tool_index"] = target_index
            episode.attrs["target_tool_name"] = tool_name
            episode.attrs["language_instruction"] = CANONICAL_INSTRUCTION.format(tool_name=tool_name)
        data.attrs["target_order"] = ",".join(TOOL_DISPLAY_NAMES)
        data.attrs["recording_profile"] = "actions_states_low_dim_observations"
        data.attrs["excluded_observation_terms"] = (
            "external_camera_1_rgb,external_camera_2_rgb,wrist_camera_rgb"
        )
        data.attrs["camera_video_source"] = "re-rendered_from_recorded_actions"
        stream.flush()
    return len(episode_names)


if __name__ == "__main__":
    # Reset events use this marker to map successful episode number to target
    # identity. Retries therefore retain the current target and demo_0..3 are
    # screwdriver, wrench, hammer, and pallet knife in that order.
    os.environ["BIN_PICKPLACE_RECORDING"] = "1"
    return_code = launch_stock_tool("scripts/tools/record_demos.py")
    if return_code == 0 and not any(argument in {"-h", "--help"} for argument in sys.argv[1:]):
        dataset_path = _dataset_path(sys.argv[1:])
        annotated_count = annotate_dataset(dataset_path)
        if annotated_count:
            print(f"Annotated {annotated_count} Bin-PickPlace episodes in {dataset_path}", flush=True)
    raise SystemExit(return_code)
