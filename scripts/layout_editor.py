"""Bin-PickPlace 场景布局编辑器。

启动完整的单环境场景后暂停物理仿真。用户可在 Isaac Sim 的 Stage/Viewport 中
用移动和旋转 gizmo 调整四件工具，然后按 F8 保存，或按 F9 保存并退出。
保存结果只写入 JSON，不会直接改动作、reset 或任务配置。

示例：

    python scripts/layout_editor.py --output datasets/manual_layout.json
"""

from __future__ import annotations

import argparse
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Bin-PickPlace-Franka-Play-v0")
parser.add_argument("--output", type=Path, default=Path("datasets/manual_layout.json"))
parser.add_argument(
    "--test-export",
    action="store_true",
    help="不进入人工编辑循环，立即导出当前布局；用于自动化检查。",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.headless and not args.test_export:
    parser.error("人工布局编辑器不能使用 --headless；自动检查时请同时传 --test-export")

# 场景包含三路相机。普通编辑模式也统一启用相机扩展，避免任务观测配置不完整。
args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import carb.input
import gymnasium as gym
import omni.appwindow
import omni.timeline
import omni.usd
import torch
from pxr import Usd, UsdGeom

import bin_pickplace_tasks  # noqa: F401, E402  # 注册 Gym ID
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from bin_pickplace_tasks.tasks.bin_pick_place.mdp.events import TOOL_NAMES  # noqa: E402
from bin_pickplace_tasks.tasks.bin_pick_place.mdp.geometry import (  # noqa: E402
    TOOL_BOUNDS_MAX,
    TOOL_BOUNDS_MIN,
    RECOMMENDED_PAIR_AISLE_XY,
    RECOMMENDED_WALL_CLEARANCE_XY,
    WHITE_INTERIOR_MAX,
    WHITE_INTERIOR_MIN,
    box_corners,
    quat_apply,
    quat_apply_inverse,
)


TOOL_PRIM_NAMES = {
    "screwdriver": "Screwdriver",
    "wrench": "Wrench",
    "hammer": "Hammer",
    "pallet_knife": "PalletKnife",
}
ENV_ROOT = "/World/envs/env_0"
WHITE_BIN_PATH = f"{ENV_ROOT}/WhiteBin"
TOOL_PATHS = {name: f"{ENV_ROOT}/{prim_name}" for name, prim_name in TOOL_PRIM_NAMES.items()}

# Isaac Sim's default NVIDIA Sans font does not contain Chinese glyphs.  Use a
# system CJK font explicitly for omni.ui widgets.  Keep an English fallback so
# the editor remains readable on machines that do not install either font.
CJK_FONT_CANDIDATES = (
    Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
)
CJK_FONT_PATH = next((path for path in CJK_FONT_CANDIDATES if path.is_file()), None)
UI_FONT_STYLE = {"font": str(CJK_FONT_PATH)} if CJK_FONT_PATH is not None else {}


def _ui_text(chinese: str, english: str) -> str:
    """Use Chinese only when a font with CJK glyphs is available."""
    return chinese if CJK_FONT_PATH is not None else english


TOOL_LABELS = {
    "screwdriver": _ui_text("螺丝刀 screwdriver", "Screwdriver"),
    "wrench": _ui_text("扳手 wrench", "Wrench"),
    "hammer": _ui_text("锤子 hammer", "Hammer"),
    "pallet_knife": _ui_text("调色刀 pallet knife", "Pallet knife"),
}


def _world_pose(stage, prim_path: str) -> tuple[list[float], list[float]]:
    """从 USD Stage 读取世界位姿，四元数返回 wxyz。"""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Stage 中找不到 Prim：{prim_path}")
    matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    translation = matrix.ExtractTranslation()
    rotation = matrix.ExtractRotationQuat()
    imaginary = rotation.GetImaginary()
    position = [round(float(value), 6) for value in translation]
    quaternion = [
        round(float(rotation.GetReal()), 7),
        round(float(imaginary[0]), 7),
        round(float(imaginary[1]), 7),
        round(float(imaginary[2]), 7),
    ]
    return position, quaternion


def _layout_diagnostics(white_pose: dict, tool_poses: dict) -> dict:
    """检查工具是否在白箱内，并报告墙距和工具间通道。"""
    white_position = torch.tensor(white_pose["position"], dtype=torch.float64)
    white_quaternion = torch.tensor(white_pose["quaternion_wxyz"], dtype=torch.float64)
    interior_min = torch.tensor(WHITE_INTERIOR_MIN, dtype=torch.float64)
    interior_max = torch.tensor(WHITE_INTERIOR_MAX, dtype=torch.float64)
    errors: list[str] = []
    tool_aabbs: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    wall_clearances: dict[str, float] = {}

    for index, name in enumerate(TOOL_NAMES):
        pose = tool_poses[name]
        position = torch.tensor([pose["position"]], dtype=torch.float64)
        quaternion = torch.tensor([pose["quaternion_wxyz"]], dtype=torch.float64)
        bounds_min = torch.tensor([TOOL_BOUNDS_MIN[index]], dtype=torch.float64)
        bounds_max = torch.tensor([TOOL_BOUNDS_MAX[index]], dtype=torch.float64)
        corners = box_corners(bounds_min, bounds_max)
        corners_world = position[:, None, :] + quat_apply(quaternion[:, None, :].expand(-1, 8, -1), corners)
        corners_white = quat_apply_inverse(
            white_quaternion[None, None, :].expand(-1, 8, -1),
            corners_world - white_position[None, None, :],
        )[0]
        aabb_min = corners_white.amin(dim=0)
        aabb_max = corners_white.amax(dim=0)
        tool_aabbs[name] = (aabb_min, aabb_max)

        if not bool(torch.all((corners_white > interior_min) & (corners_white < interior_max))):
            errors.append(f"{name} 没有完全位于白箱保守内区")

        margins = torch.cat((aabb_min[:2] - interior_min[:2], interior_max[:2] - aabb_max[:2]))
        wall_clearance = float(torch.min(margins))
        wall_clearances[name] = round(wall_clearance, 6)
        if wall_clearance < RECOMMENDED_WALL_CLEARANCE_XY:
            errors.append(
                f"{name} 的最小水平墙距为 {wall_clearance:.4f} m，"
                f"低于建议值 {RECOMMENDED_WALL_CLEARANCE_XY:.3f} m"
            )

    pair_clearances: dict[str, float] = {}
    names = list(TOOL_NAMES)
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            left_min, left_max = tool_aabbs[left_name]
            right_min, right_max = tool_aabbs[right_name]
            horizontal_gaps = torch.maximum(right_min[:2] - left_max[:2], left_min[:2] - right_max[:2])
            approach_aisle = float(torch.max(horizontal_gaps))
            pair_key = f"{left_name}__{right_name}"
            pair_clearances[pair_key] = round(approach_aisle, 6)
            if approach_aisle < RECOMMENDED_PAIR_AISLE_XY:
                errors.append(
                    f"{left_name} 与 {right_name} 的最大水平分离通道为 {approach_aisle:.4f} m，"
                    f"低于建议值 {RECOMMENDED_PAIR_AISLE_XY:.3f} m"
                )

    return {
        "valid": not errors,
        "errors": errors,
        "wall_clearance_xy_m": wall_clearances,
        "pair_approach_aisle_xy_m": pair_clearances,
        "recommended_wall_clearance_xy_m": RECOMMENDED_WALL_CLEARANCE_XY,
        "recommended_pair_approach_aisle_xy_m": RECOMMENDED_PAIR_AISLE_XY,
    }


def _capture_layout(stage) -> dict:
    white_position, white_quaternion = _world_pose(stage, WHITE_BIN_PATH)
    white_pose = {"position": white_position, "quaternion_wxyz": white_quaternion}
    tool_poses = {}
    for name, prim_path in TOOL_PATHS.items():
        position, quaternion = _world_pose(stage, prim_path)
        tool_poses[name] = {
            "prim_path": prim_path,
            "position": position,
            "quaternion_wxyz": quaternion,
        }
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": args.task,
        "frame": "world; env_0 origin is zero",
        "quaternion_order": "wxyz",
        "editor_note": "人工布局候选；需通过几何测试和 CUDA 落稳验证后才能写入正式配置。",
        "white_bin": {"prim_path": WHITE_BIN_PATH, **white_pose},
        "tools": tool_poses,
        "diagnostics": _layout_diagnostics(white_pose, tool_poses),
    }


def _write_layout(stage, output_path: Path) -> dict:
    payload = _capture_layout(stage)
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[Bin-PickPlace 布局编辑器] 已保存：{output_path}", flush=True)
    for name, pose in payload["tools"].items():
        print(
            f"  {name}: position={pose['position']}, quaternion_wxyz={pose['quaternion_wxyz']}",
            flush=True,
        )
    diagnostics = payload["diagnostics"]
    if diagnostics["valid"]:
        print("  静态布局检查：通过", flush=True)
    else:
        print("  静态布局检查：存在需要复核的项目", flush=True)
        for error in diagnostics["errors"]:
            print(f"    - {error}", flush=True)
    return payload


def main() -> None:
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    cfg.scene.num_envs = 1
    env = gym.make(args.task, cfg=cfg).unwrapped
    env.reset()

    timeline = omni.timeline.get_timeline_interface()
    timeline.pause()
    # 让暂停状态、Stage 树和视口 gizmo 完成一次 UI 更新。
    simulation_app.update()
    stage = omni.usd.get_context().get_stage()

    if args.test_export:
        # Exercise the same font/style path in automated runs, even though a
        # headless test cannot visually inspect glyphs.
        import omni.ui as ui

        font_probe_window = ui.Window("Bin-PickPlace Font Probe", width=320, height=80)
        with font_probe_window.frame:
            ui.Label(
                _ui_text("中文字体加载检查", "CJK font fallback check"),
                style=UI_FONT_STYLE,
            )
        simulation_app.update()
        font_probe_window.visible = False
        _write_layout(stage, args.output)
        env.close()
        return

    should_exit = False
    status_label = None

    def select_tool(name: str) -> None:
        omni.usd.get_context().get_selection().set_selected_prim_paths([TOOL_PATHS[name]], True)
        if status_label is not None:
            status_label.text = _ui_text(f"已选择：{TOOL_LABELS[name]}", f"Selected: {TOOL_LABELS[name]}")

    def save_layout(exit_after: bool = False) -> None:
        nonlocal should_exit
        try:
            payload = _write_layout(stage, args.output)
            if status_label is not None:
                if payload["diagnostics"]["valid"]:
                    state = _ui_text("通过静态检查", "Static checks passed")
                else:
                    state = _ui_text("已保存，但存在检查警告", "Saved with validation warnings")
                status_label.text = f"{state}: {args.output.expanduser().resolve()}"
            if exit_after:
                should_exit = True
        except Exception as error:
            traceback.print_exc()
            if status_label is not None:
                status_label.text = _ui_text(f"保存失败：{error}", f"Save failed: {error}")

    import omni.ui as ui

    # Kit's native window title bar cannot take a per-window font override, so
    # keep the title ASCII and apply the CJK font to every content widget.
    window = ui.Window("Bin-PickPlace Layout Editor", width=500, height=390)
    with window.frame:
        with ui.VStack(spacing=6):
            ui.Label(
                _ui_text("物理已暂停。请不要点击 Play。", "Physics is paused. Do not click Play."),
                style={**UI_FONT_STYLE, "color": 0xFF66CCFF},
            )
            ui.Label(
                _ui_text(
                    "先点击下面的工具按钮，再在视口使用移动/旋转 gizmo。建议只调整 X、Y 和绕 Z 的朝向，保留 Z≈0.020 m。",
                    "Select a tool, then use the viewport move/rotate gizmo. Adjust X, Y and yaw; keep Z near 0.020 m.",
                ),
                word_wrap=True,
                height=55,
                style=UI_FONT_STYLE,
            )
            ui.Separator()
            ui.Label(_ui_text("选择工具：", "Select a tool:"), style=UI_FONT_STYLE)
            for tool_name in TOOL_NAMES:
                ui.Button(
                    TOOL_LABELS[tool_name],
                    clicked_fn=lambda name=tool_name: select_tool(name),
                    height=30,
                    style=UI_FONT_STYLE,
                )
            ui.Separator()
            ui.Label(
                _ui_text("F8：保存 JSON 并继续编辑；F9：保存并退出。", "F8: save JSON; F9: save and exit."),
                word_wrap=True,
                height=30,
                style=UI_FONT_STYLE,
            )
            ui.Button(
                _ui_text("保存位姿（F8）", "Save poses (F8)"),
                clicked_fn=lambda: save_layout(False),
                height=34,
                style=UI_FONT_STYLE,
            )
            ui.Button(
                _ui_text("保存并退出（F9）", "Save and exit (F9)"),
                clicked_fn=lambda: save_layout(True),
                height=34,
                style=UI_FONT_STYLE,
            )
            status_label = ui.Label(
                _ui_text("输出：", "Output: ") + str(args.output.expanduser().resolve()),
                word_wrap=True,
                height=45,
                style=UI_FONT_STYLE,
            )

    app_window = omni.appwindow.get_default_app_window()
    input_interface = carb.input.acquire_input_interface()
    keyboard = app_window.get_keyboard()

    def on_keyboard_event(event, *_unused) -> bool:
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name == "F8":
                save_layout(False)
            elif event.input.name == "F9":
                save_layout(True)
        return True

    keyboard_subscription = input_interface.subscribe_to_keyboard_events(keyboard, on_keyboard_event)
    print("\n[Bin-PickPlace 布局编辑器] 已暂停物理仿真。", flush=True)
    print("  1. 在浮动窗口选择工具。", flush=True)
    print("  2. 用视口 gizmo 调整 X/Y 和 yaw；建议保留 Z≈0.020 m。", flush=True)
    print("  3. 按 F8 保存，或按 F9 保存并退出。", flush=True)
    print(f"  输出文件：{args.output.expanduser().resolve()}\n", flush=True)
    if CJK_FONT_PATH is not None:
        print(f"  UI 中文字体：{CJK_FONT_PATH}", flush=True)
    else:
        print("  未找到 CJK 字体，浮动窗口已自动切换为英文。", flush=True)

    try:
        while simulation_app.is_running() and not should_exit:
            simulation_app.update()
    finally:
        input_interface.unsubscribe_to_keyboard_events(keyboard, keyboard_subscription)
        # 持有 window 引用直到循环结束，避免 Kit 提前销毁面板。
        window.visible = False
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
