"""Keyboard SE(3) teleoperation entry point."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Bin-PickPlace-Franka-Play-v0")
parser.add_argument(
    "--sensitivity",
    type=float,
    default=1.0,
    help="Bin-PickPlace keyboard speed multiplier (0 < value <= 5; default: 1.0).",
)
args, unknown = parser.parse_known_args()
if not 0.0 < args.sensitivity <= 5.0:
    parser.error("--sensitivity must be greater than 0 and no greater than 5")

# Keep behavior aligned with Isaac Lab's maintained teleoperation tool.
isaaclab_root_value = os.environ.get("ISAACLAB_ROOT")
if not isaaclab_root_value:
    raise SystemExit("Set ISAACLAB_ROOT to your IsaacLab checkout before running teleop.py")
isaaclab_root = Path(isaaclab_root_value).expanduser().resolve()
if not (isaaclab_root / "isaaclab.sh").is_file():
    raise SystemExit(f"ISAACLAB_ROOT does not contain isaaclab.sh: {isaaclab_root}")

# The bridge runs inside Isaac Lab's Python process and imports bin_pickplace_tasks
# before handing control to the maintained stock teleoperation program.
bridge = Path(__file__).with_name("isaaclab_teleop_bridge.py").resolve()
forwarded = list(unknown)
if "--enable_cameras" not in forwarded:
    forwarded.append("--enable_cameras")

# Bin-PickPlace registers a named keyboard device in the task configuration.  Isaac
# Lab therefore does not apply its fallback ``--sensitivity`` value.  Pass the
# requested multiplier explicitly to the task configuration instead.
child_env = {
    **os.environ,
    "ISAACLAB_ROOT": str(isaaclab_root),
    "BIN_PICKPLACE_TELEOP_SENSITIVITY": str(args.sensitivity),
}
print(
    f"[Bin-PickPlace] keyboard sensitivity x{args.sensitivity:g} "
    f"(translation command speed approximately {0.36 * args.sensitivity:.2f} m/s)",
    flush=True,
)

raise SystemExit(
    subprocess.call(
        [
            # Preserve the activated Isaac Lab interpreter in which this
            # external package was installed.  ``isaaclab.sh -p`` may select
            # Isaac Sim's bundled Python and lose the editable installation.
            sys.executable,
            str(bridge),
            "--task",
            args.task,
            "--teleop_device",
            "keyboard",
            "--sensitivity",
            str(args.sensitivity),
            *forwarded,
        ],
        env=child_env,
    )
)
