"""Compare tool masses in metadata, scene configuration, USD, and PhysX runtime.

Example:

    python scripts/inspect_tool_masses.py --headless --enable_cameras
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Bin-PickPlace-Franka-Play-v0")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not args.enable_cameras:
    parser.error("This environment has camera observations; add --enable_cameras")

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
from pxr import Usd, UsdPhysics

import bin_pickplace_tasks  # noqa: F401, E402  # register Gym environments
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


TOOL_SPECS = {
    "screwdriver": ("screwdriver", "physics_69d7449cf508477722c7054a.json"),
    "wrench": ("wrench", "physics_69d9fec9f088e31d5f555664.json"),
    "hammer": ("hammer", "physics_69d9ff4af088e31d5f55843b.json"),
    "pallet_knife": ("pallet knife", "physics_69e6ffc8518f642e4d86b8b4.json"),
}


def authored_usd_masses(usd_path: Path) -> list[tuple[str, float]]:
    """Return prims with explicitly authored MassAPI values in a USD file."""
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"Unable to open USD: {usd_path}")
    result = []
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.MassAPI):
            value = UsdPhysics.MassAPI(prim).GetMassAttr().Get()
            if value is not None:
                result.append((str(prim.GetPath()), float(value)))
    return result


def main() -> None:
    env = None
    try:
        cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
        rows = []
        for asset_name, (display_name, metadata_filename) in TOOL_SPECS.items():
            asset_cfg = getattr(cfg.scene, asset_name)
            usd_path = Path(asset_cfg.spawn.usd_path)
            metadata_path = usd_path.parent / metadata_filename
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata_mass = metadata.get("total_mass", metadata.get("mass"))
            rows.append(
                {
                    "asset_name": asset_name,
                    "display_name": display_name,
                    "metadata_mass": float(metadata_mass),
                    "configured_mass": float(asset_cfg.spawn.mass_props.mass),
                    "usd_path": usd_path,
                    "usd_masses": authored_usd_masses(usd_path),
                }
            )

        env = gym.make(args.task, cfg=cfg).unwrapped
        env.reset()
        print("\nTool mass verification (kg)")
        print("-" * 96)
        for row in rows:
            runtime_mass = float(env.scene[row["asset_name"]].data.default_mass[0, 0].item())
            row["runtime_mass"] = runtime_mass
            usd_text = ", ".join(f"{path}={mass:g}" for path, mass in row["usd_masses"]) or "not authored"
            print(
                f'{row["display_name"]:<12} metadata={row["metadata_mass"]:<6g} '
                f'configured={row["configured_mass"]:<6g} PhysX={runtime_mass:<6g} USD={usd_text}'
            )
            if abs(runtime_mass - row["configured_mass"]) > 1.0e-6:
                raise AssertionError(
                    f'{row["display_name"]} runtime mass {runtime_mass} does not match '
                    f'configured mass {row["configured_mass"]}'
                )
        print("-" * 96)
        print("Result: PhysX runtime masses match the scene configuration, which is authoritative.")
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
