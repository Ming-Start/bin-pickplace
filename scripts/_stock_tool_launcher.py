"""Shared launcher for Isaac Lab tools used by this external project."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def launch_stock_tool(relative_tool_path: str) -> int:
    """Run a stock Isaac Lab tool after external-task registration."""
    isaaclab_root_value = os.environ.get("ISAACLAB_ROOT")
    if not isaaclab_root_value:
        raise SystemExit("Set ISAACLAB_ROOT to your IsaacLab checkout before running this script")

    isaaclab_root = Path(isaaclab_root_value).expanduser().resolve()
    stock_tool = (isaaclab_root / relative_tool_path).resolve()
    if not (isaaclab_root / "isaaclab.sh").is_file():
        raise SystemExit(f"ISAACLAB_ROOT does not contain isaaclab.sh: {isaaclab_root}")
    if isaaclab_root not in stock_tool.parents or not stock_tool.is_file():
        raise SystemExit(f"Isaac Lab tool not found: {stock_tool}")

    forwarded = list(sys.argv[1:])
    if "--enable_cameras" not in forwarded:
        forwarded.append("--enable_cameras")

    bridge = Path(__file__).with_name("isaaclab_tool_bridge.py").resolve()
    child_env = {
        **os.environ,
        "ISAACLAB_ROOT": str(isaaclab_root),
        "BIN_PICKPLACE_STOCK_TOOL": str(stock_tool),
    }
    # Keep the interpreter in which ``bin-pickplace-tasks`` was installed.  Calling
    # ``isaaclab.sh -p`` here can silently switch to Isaac Sim's bundled Python,
    # where an editable install made in the active Isaac Lab environment is not
    # visible.
    return subprocess.call([sys.executable, str(bridge), *forwarded], env=child_env)
