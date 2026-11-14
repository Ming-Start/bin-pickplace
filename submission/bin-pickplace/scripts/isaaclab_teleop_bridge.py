"""Register Bin-PickPlace tasks, then execute Isaac Lab's maintained teleop tool.

This file is launched with the active Isaac Lab Python by :mod:`scripts.teleop`.
Keeping the registration import in the simulator subprocess is essential:
Isaac Lab's stock tool imports ``isaaclab_tasks`` but cannot know about an
independently installed external project.
"""

from __future__ import annotations

import os
import runpy
from pathlib import Path

import bin_pickplace_tasks  # noqa: F401  # registers Bin-PickPlace-* Gym environments


isaaclab_root = Path(os.environ["ISAACLAB_ROOT"]).expanduser().resolve()
stock_teleop = isaaclab_root / "scripts/environments/teleoperation/teleop_se3_agent.py"
if not stock_teleop.is_file():
    raise SystemExit(f"Isaac Lab teleoperation tool not found: {stock_teleop}")

runpy.run_path(str(stock_teleop), run_name="__main__")
