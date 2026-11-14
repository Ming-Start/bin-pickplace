"""Run Isaac Lab replay with state-shape compatibility and target restoration."""

import os
import sys

from _stock_tool_launcher import launch_stock_tool


if __name__ == "__main__":
    if "--validate_success_rate" in sys.argv[1:]:
        raise SystemExit(
            "Stock replay calls a persistent success term only once at episode end. "
            "Use scripts/export_demo_videos.py for target-aware, per-frame metric validation."
        )
    os.environ["BIN_PICKPLACE_REPLAY"] = "1"
    raise SystemExit(launch_stock_tool("scripts/tools/replay_demos.py"))
