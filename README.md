# Isaac Lab 箱体抓取放置（Bin Pick-and-Place）环境

本仓库是一个独立、可安装的 Isaac Lab manager-based 环境，用于语言条件控制的 Franka Panda 键盘遥操作。操作者需要按照指令从白色源箱中拿起指定工具，并将其释放在蓝色目标箱内。两台外部 RGB 相机与一台腕部 RGB 相机同步观测整个过程。

## 完成状态

环境、可移植资产、7 维相对差分 IK 动作、字典观测、重置逻辑、目标感知任务指标、相机流水线、遥操作/录制/回放封装、验证工具、人工演示与最终视频均已完成。

| Episode | 目标工具 | 动作数 / 视频帧数 | 时长 |
| --- | --- | ---: | ---: |
| `demo_0` | 螺丝刀（screwdriver） | 2129 | 70.97 s |
| `demo_1` | 扳手（wrench） | 3521 | 117.37 s |
| `demo_2` | 锤子（hammer） | 2173 | 72.43 s |
| `demo_3` | 油漆刮刀（pallet knife） | 2122 | 70.73 s |

全部最终视频均为 H.264/yuv420p、1280x1024、30 fps，帧数与对应 HDF5 动作数完全一致。每条动作重放均到达了环境指标检测的成功条件。

## 环境要求

- Isaac Sim 5.1
- 一份兼容且未修改的 Isaac Lab 检出（已在 Isaac Lab 0.54.3 上验证）
- 激活 Isaac Lab 环境中的 Python 3.11
- 支持 CUDA 的 GPU（用于相机、遥操作与视频导出）

## 安装

在仓库根目录执行：

```bash
python -m pip install --no-build-isolation -e source/bin_pickplace_tasks
python scripts/validate_install.py
python scripts/list_envs.py
```

注册的环境 ID：

- `Bin-PickPlace-Franka-v0`：多环境配置，随机采样目标。
- `Bin-PickPlace-Franka-Play-v0`：确定性的单环境评估配置。

## 环境契约

- 场景：Franka Panda、Isaac Lab 自带 Seattle Lab 桌子、白色源箱、蓝色目标箱与四件指定工具。
- 指令：`Pick the <tool_name> from the white bin and place it in the blue bin.`
- 动作：`[dx, dy, dz, dRx, dRy, dRz, gripper]`。
- 机械臂控制：相对 6 自由度阻尼最小二乘差分逆运动学（differential IK）。
- 夹爪控制：一个二值动作。
- 物理：120 Hz，控制降采样（decimation）为 4。
- 控制、渲染、观测与相机：30 Hz 同步。
- 相机：两路外部 RGB 视角与一路腕部 RGB 视角，均为 640x480。
- 成功条件：被指令目标的八个局部包围盒角点全部位于蓝色目标箱内部、目标已稳定、夹爪张开并分离，且上述条件连续保持十个控制步。
- 终止条件：成功、不可恢复的目标丢失与超时是相互独立的终止项。

详细的坐标系、阈值、相机标定、设计权衡、验证过程与 AI 使用披露见 `REPORT.md`。

## 零智能体验证

要求的零智能体（zero-agent）入口：

```bash
python scripts/zero_agent.py \
  --task Bin-PickPlace-Franka-Play-v0 \
  --headless --enable_cameras --steps 300
```

## 键盘遥操作

将 `ISAACLAB_ROOT` 指向兼容的 Isaac Lab 检出：

```bash
export ISAACLAB_ROOT=/absolute/path/to/IsaacLab
python scripts/teleop.py \
  --task Bin-PickPlace-Franka-Play-v0 \
  --sensitivity 1.0
```

该封装保留当前 Python 解释器，在启动 Isaac Lab 原生工具之前导入 `bin_pickplace_tasks`，并自动启用相机。

灵敏度为 `1.0` 时，每个 30 Hz 控制步的名义平移指令约为 `0.012 m`（`0.36 m/s`）。接近 `2.0` 的值适合自由空间移动；`0.4-0.8` 推荐用于最终对准与接触阶段。允许范围为 `(0, 5]`。

按 `R` 重置。确定性环境会打印当前目标的标准指令。

## 人工演示录制

最终数据集已包含在仓库中。如需重新录制，请先保存或移走现有文件，然后运行：

```bash
export ISAACLAB_ROOT=/absolute/path/to/IsaacLab
python scripts/record_demos.py \
  --task Bin-PickPlace-Franka-Play-v0 \
  --teleop_device keyboard \
  --dataset_file datasets/bin_pickplace_demos.hdf5 \
  --num_demos 4 --num_success_steps 10
```

成功的 episode 按标准顺序分配目标：

```text
screwdriver -> wrench -> hammer -> pallet knife
```

录制 profile 保存初始状态、逐步场景状态、原始与处理后动作，以及低维策略观测。它不会在 episode 缓冲中累积三路 480x640 RGB 张量，从而避免长演示耗尽 GPU 显存。RGB 由录制的动作在最终视频导出时重新渲染，这一偏差已记录在 HDF5 元数据中。

## 回放

可选的原生状态/动作回放：

```bash
export ISAACLAB_ROOT=/absolute/path/to/IsaacLab
python scripts/replay_demos.py \
  --task Bin-PickPlace-Franka-Play-v0 \
  --dataset_file datasets/bin_pickplace_demos.hdf5 \
  --validate_states
```

兼容性封装仅移除原生 `EpisodeData.get_state()` 路径引入的合成前导单例维度，并在重置后根据每个 episode 的 HDF5 元数据恢复目标身份。原有的逐元素 `0.01` 比较容差保持不变。

不要对本任务使用原生的 `--validate_success_rate`。原生工具只在 episode 结束时调用一次有状态的持续成功项，无法验证一个十帧条件。逐帧的目标感知成功验证由视频导出器完成。

## 三视角视频导出

重新生成现有视频：

```bash
python scripts/export_demo_videos.py \
  --task Bin-PickPlace-Franka-Play-v0 \
  --dataset-file datasets/bin_pickplace_demos.hdf5 \
  --output-dir datasets/videos \
  --headless --enable_cameras --overwrite
```

导出器恢复每条录制的初始状态、只重放人工动作、渲染三路同步视角、叠加相机标签与语言指令、每步评估成功状态，并且只有在目标感知指标触发后才提交最终 MP4。

最终交付文件：

```text
datasets/videos/demo_0.mp4
datasets/videos/demo_1.mp4
datasets/videos/demo_2.mp4
datasets/videos/demo_3.mp4
```

`demo_1` 开头包含约 24 秒的初始空闲时间，并在 120 秒超时前 2.63 秒完成。它是有效的，但不如其他演示精炼。

## 验证

CPU 测试：

```bash
python -m unittest -v \
  tests.test_metric_geometry \
  tests.test_recording_profile \
  tests.test_replay_profile
```

CUDA 冒烟测试：

```bash
python scripts/smoke_test.py \
  --task Bin-PickPlace-Franka-Play-v0 \
  --headless --enable_cameras --steps 240 \
  --capture-dir datasets/validation/hybrid_collision_final
```

实时任务指标探针：

```bash
python scripts/probe_task_metrics.py \
  --task Bin-PickPlace-Franka-Play-v0 \
  --headless --enable_cameras
```

质量与持载探针：

```bash
python scripts/inspect_tool_masses.py --headless --enable_cameras

python scripts/probe_arm_load_holding.py \
  --headless --enable_cameras --steps 180 \
  --max-drop 0.006 --max-angle 0.02
```

## 可移植资产与打包

运行时资产存放在 `source/bin_pickplace_tasks/bin_pickplace_tasks/assets` 下，并相对已安装的包路径解析。没有任何运行时代码或打包资产依赖开发机的用户主目录或原始下载位置。

干净的提交副本排除了原始下载文件、过期构建产物、egg-info、缓存、布局编辑器候选文件与历史验证截图，保留了可安装源码、可移植运行时资产、所需脚本、测试、中文文档、HDF5 加分数据集、四个必需的 MP4 视频与最新验证图片。
