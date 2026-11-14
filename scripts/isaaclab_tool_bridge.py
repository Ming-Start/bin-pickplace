"""Register Bin-PickPlace environments, then execute a selected stock Isaac Lab tool."""

from __future__ import annotations

import os
import runpy
from pathlib import Path

import bin_pickplace_tasks  # noqa: F401  # registers Bin-PickPlace-* Gym environments


isaaclab_root = Path(os.environ["ISAACLAB_ROOT"]).expanduser().resolve()
stock_tool = Path(os.environ["BIN_PICKPLACE_STOCK_TOOL"]).expanduser().resolve()
if isaaclab_root not in stock_tool.parents or not stock_tool.is_file():
    raise SystemExit(f"Refusing invalid Isaac Lab tool path: {stock_tool}")


def _run_recording_tool_with_post_app_patch() -> None:
    """Execute the stock recorder after injecting one post-AppLauncher hook."""
    from recording_profile import inject_profile_after_app_launcher

    source = stock_tool.read_text(encoding="utf-8")
    patched_source = inject_profile_after_app_launcher(source)
    namespace = {
        "__name__": "__main__",
        "__file__": str(stock_tool),
        "__package__": None,
    }
    exec(compile(patched_source, str(stock_tool), "exec"), namespace)


def _run_replay_tool_with_post_app_patch() -> None:
    """Inject Bin-PickPlace state-shape compatibility and per-episode target restoration."""
    from replay_profile import inject_profile

    source = stock_tool.read_text(encoding="utf-8")
    patched_source = inject_profile(source)
    namespace = {
        "__name__": "__main__",
        "__file__": str(stock_tool),
        "__package__": None,
    }
    exec(compile(patched_source, str(stock_tool), "exec"), namespace)


if os.environ.get("BIN_PICKPLACE_RECORDING") == "1":
    _run_recording_tool_with_post_app_patch()
elif os.environ.get("BIN_PICKPLACE_REPLAY") == "1":
    _run_replay_tool_with_post_app_patch()
else:
    runpy.run_path(str(stock_tool), run_name="__main__")
