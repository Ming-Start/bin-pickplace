# 作业题目解释、实现与交付说明

## 1. 这个题目是什么

这是一个“机器人仿真环境工程”作业，不是训练一个能自动完成任务的策略。

你要交付的是一个独立、可安装的 Isaac Lab 外部项目。安装后，它注册一个 manager-based、
single-agent Gym 环境：Franka Panda 根据一句语言指令，从白色源箱里拿起指定工具，再把它放进
蓝色目标箱。

场景必须包含：

- Franka Panda；
- Isaac Lab 自带 Seattle Lab table；
- 下载资产中的白箱、蓝箱；
- screwdriver、wrench、hammer、pallet knife 四件工具；
- 两个外部 RGB 相机和一个腕部 RGB 相机；
- 键盘控制的 7 维动作：
  `[dx, dy, dz, dRx, dRy, dRz, gripper]`。

题目还要求四条真实成功演示视频：四件工具各一条，三相机同步拼接，显示语言指令，并且视频必须
一直录到代码中的成功指标真正触发。HDF5 录制和可重放数据属于加分/强化证据。

当前上述工程和人工演示均已完成：`datasets/bin_pickplace_demos.hdf5` 含四条成功 episode，
`datasets/videos/demo_0.mp4` 到 `demo_3.mp4` 均已通过完整动作重放、逐帧指标验证与 H.264 编码检查。

## 2. 怎么理解题目

这道题真正考的是“把自然语言任务变成可靠的工程契约”，可以分成六层理解。

### 2.1 可安装，而不是只能在作者电脑运行

不能修改 Isaac Lab 核心代码，不能把 `/home/某用户名` 或 Google Drive 下载目录写死在配置里。
`python -m pip install -e source/bin_pickplace_tasks` 之后，换一台兼容机器也应能注册并启动环境。

### 2.2 场景正确，而不是只要能渲染

桌子、箱子和工具必须在物理上对齐；工具 reset 后要在白箱里、不相交、不持续弹跳，速度也必须清零。
原始白箱只有约 29.8 x 14.1 cm，而 hammer 长约 30.9 cm，四件工具根本无法按原尺寸同时无重叠摆入。
所以本实现保留指定白箱资产，只把 XY 平面放大 2.4 倍，并用布局编辑器在完整场景中人工设计四个单层位姿。旋转 AABB 的墙距和通道只作为保守提示；最终以真实 PhysX 碰撞体落稳后的白箱内含、速度和 XY 漂移为准。

### 2.3 语言、目标和指标必须是同一个对象

每局的语言字符串、观测 one-hot、目标相对位置、成功判定和失败判定都必须引用同一个目标索引。
否则“指令让拿扳手，但把螺丝刀放进蓝箱也成功”就是严重错误。

规范指令是：

`Pick the <tool_name> from the white bin and place it in the blue bin.`

### 2.4 “放进箱子”必须形式化

仅判断工具 root 在蓝箱范围内不够：长工具可能横跨箱沿，但 root 正好在中央。

本实现把目标工具实测 OBB 的 8 个角点全部变换到蓝箱局部坐标系。只有所有角点都在保守内区、
线速度和角速度足够小、夹爪已张开并远离工具，而且连续 10 个控制帧都成立，才判成功。

因此下列情况都必须失败：

- 放进去的是错误工具；
- 正确工具仍夹在手里；
- 工具仍在移动、弹跳或旋转；
- 工具搭在箱沿；
- 条件只短暂满足，没有连续稳定。

### 2.5 相机是观测接口，不只是截图

三路相机必须有稳定 observation key、相同的 `640x480` RGB 形状、30 Hz 更新频率，并与 30 Hz
控制步对齐。两个外部视角负责覆盖双箱和搬运路径，腕部相机负责接近和抓取局部。

### 2.6 代码完成不等于作业全部完成

最终证据必须包括真实人工键盘演示。自动写一个脚本直接搬动物体或伪造视频，不符合题意。
所以本仓库把“自动可完成的工程部分”和“必须由人完成的交互证据”明确分开。

## 3. 我该怎么做

### 第一步：安装和基础注册

```bash
python -m pip install --no-build-isolation -e source/bin_pickplace_tasks
python scripts/validate_install.py
python scripts/list_envs.py
```

应看到：

- `Bin-PickPlace-Franka-v0`
- `Bin-PickPlace-Franka-Play-v0`

### 第二步：跑自动验收

先跑题目明确要求的 zero-agent 入口：

```bash
python scripts/zero_agent.py \
  --task Bin-PickPlace-Franka-Play-v0 \
  --headless --enable_cameras --steps 300
```

再跑断言更严格的测试：

```bash
python -m unittest -v \
  tests.test_metric_geometry \
  tests.test_recording_profile \
  tests.test_replay_profile
```

```bash
python scripts/smoke_test.py \
  --task Bin-PickPlace-Franka-Play-v0 \
  --headless --enable_cameras --steps 240 \
  --capture-dir datasets/validation/manual_layout_final
```

```bash
python scripts/probe_task_metrics.py \
  --task Bin-PickPlace-Franka-Play-v0 \
  --headless --enable_cameras
```

```bash
python scripts/probe_recording_targets.py \
  --task Bin-PickPlace-Franka-Play-v0 \
  --headless --enable_cameras
```

第二、第三条需要 CUDA；如果运行环境限制 GPU，应在沙盒外执行。

### 第三步：先练习键盘控制

```bash
export ISAACLAB_ROOT=/你的/IsaacLab/绝对路径
python scripts/teleop.py --task Bin-PickPlace-Franka-Play-v0 --sensitivity 1.0
```

Isaac Lab 会打印键位。按 `R` 可重置；每次确定性 reset 都会在终端打印当前规范指令。
默认 `--sensitivity 1.0` 时，按住单个平移键约为每控制步 0.012 m（30 Hz 下理论指令速度约
0.36 m/s），旋转键约为每步 0.0375 rad。现在该参数会直接传给任务注册的键盘设备；需要快速
跨越工作区时可用 `--sensitivity 2.0`（约 0.72 m/s），抓取前精细调整可用 `0.5`。

如果需要亲自决定工具初始位姿，可以先运行：

```bash
python scripts/layout_editor.py --output datasets/manual_layout.json
```

该脚本加载完整场景后暂停物理，并提供四件工具的选择按钮。使用视口 gizmo 调整 X、Y 和 yaw，尽量
保留 `Z≈0.020 m`；按 `F8` 保存并继续，按 `F9` 保存并退出。输出 JSON 同时给出白箱内含、最小墙距
和工具间通道诊断，但它不会自动覆盖正式配置。人工调整结束后，需要根据 JSON 同步场景初始位姿与
reset 位姿，并重新进行 CPU 和 CUDA 验证。

### 第四步：录四个真实成功 episode

这一步已经完成。下面的命令用于复现或重新录制；已有最终文件时应先另存旧文件，避免误覆盖有效数据。

录制时采用低显存 profile：保留动作、初始状态、逐帧 scene state、processed action 和非图像 policy 观测；三路 RGB 不写入逐帧 GPU episode buffer，而是在最终视频导出时按动作重放并重新渲染。这避免数分钟人工操作在 `torch.stack` 阶段耗尽显存，同时保持 state replay 和三视角视频证据完整。

```bash
python scripts/record_demos.py \
  --task Bin-PickPlace-Franka-Play-v0 \
  --teleop_device keyboard \
  --dataset_file datasets/bin_pickplace_demos.hdf5 \
  --num_demos 4 --num_success_steps 10
```

录制模式会固定顺序：

1. screwdriver；
2. wrench；
3. hammer；
4. pallet knife。

一次失败后按 `R`，目标不会提前跳到下一个；只有当前 episode 被指标确认成功并导出后才前进。

### 第五步：重放并输出视频

四条最终视频已经通过 exporter 的完整动作重放和逐帧成功指标验证。stock state replay 是加分项中的附加诊断，不是生成 MP4 的前置条件：

```bash
python scripts/replay_demos.py \
  --task Bin-PickPlace-Franka-Play-v0 \
  --dataset_file datasets/bin_pickplace_demos.hdf5 \
  --validate_states
```

```bash
python scripts/export_demo_videos.py \
  --task Bin-PickPlace-Franka-Play-v0 \
  --dataset-file datasets/bin_pickplace_demos.hdf5 \
  --output-dir datasets/videos \
  --headless --enable_cameras --overwrite
```

导出器不会生成“看起来成功”的假视频。它只重放 HDF5 里真实记录的人类动作；若重放没有再次触发
实现的 metric，就报错并拒绝生成最终文件。

说明：stock replay 的 `--validate_success_rate` 只在 episode 末调用一次 success 函数，无法验证本项目
函数内部的连续 10 帧条件，也不会自动恢复 Python 层目标索引，所以包装器会明确拒绝这个参数。
`record_demos.py` 会把目标索引、工具名和规范指令写入 HDF5；逐帧、目标感知的成功验证由 exporter
完成。stock replay 仍用于动作和 scene state 重放检查。Bin-PickPlace 桥接会移除 Isaac Lab 0.54.3
`EpisodeData.get_state()` 额外添加、但比较端未保留的单例批次维度，并在每条 episode reset 后恢复
HDF5 中的目标索引和语言；它不会放宽原有 `0.01` 状态数值误差阈值。

当前最终文件为：

- `datasets/videos/demo_0.mp4`
- `datasets/videos/demo_1.mp4`
- `datasets/videos/demo_2.mp4`
- `datasets/videos/demo_3.mp4`

四条均已确认为 1280x1024、30 fps、H.264/yuv420p，包含三路视角标签、准确语言指令和 metric 状态；视频帧数分别为 2129、3521、2173、2122，与 HDF5 中对应 action 数完全一致，末尾均显示 `metric: SUCCESS (detected)`。

## 4. 已经帮你完成了什么

### 4.1 项目与场景

- 独立 `bin_pickplace_tasks` Python 包和两个 Gym 环境注册；
- Franka、Seattle table、双箱、四工具完整场景；
- 资产 root、桌面高度、箱体方向和初始姿态重新校准；
- 白箱 XY 2.4 倍缩放，以及四个实测、留有夹爪通道的单层 reset 槽位；
- robot、bins、tools 的位置和速度集中 reset；
- 普通环境随机目标，PLAY 环境可复现目标轮换。

### 4.2 控制与观测

- 6 维 relative DLS IK + 1 维二值夹爪，Action Manager 实测维度为 7；
- 零运动输入保持最后一次 TCP 目标，避免载荷下相对 IK 每帧重置目标造成整臂持续下沉；肩部/前臂刚度为 `800`、阻尼为 `80`，并保留真实 `87/12 N·m` 力矩上限；
- joints、EE、gripper、四工具、双箱 bounds、目标相对向量、目标 one-hot、previous action；
- 三路稳定命名的 `480x640x3 uint8` RGB 字典观测；
- 120 Hz physics（每个 30 Hz 动作 4 个子步）、30 Hz control/render/camera 同步；Franka 求解迭代为 `32/4`，工具接触偏移为 `0.003 m`。CPU PhysX 启用 CCD；Isaac Sim 5.1 的 CUDA dynamics 不支持 CCD，因此 GPU 使用高频子步降低穿透。
- 碰撞表示按形状混合配置：Franka 使用官方碰撞体，螺丝刀为解析胶囊杆 + 凸包手柄，扳手为 `256/6` sparse SDF，锤子和 pallet knife 保持凸分解。这样保留扳手凹口和细杆接触，同时避免给整台机器人或所有工具使用昂贵 SDF。

### 4.3 成功、失败和 timeout

- 目标工具专属判定；
- 8 个 OBB 角点全部位于蓝箱局部内区；
- 线/角速度 settled 条件；
- 张开夹爪 + TCP 距离 released 条件；
- 连续 10 帧 persistence；
- failure 与 120 秒 timeout 分离。

### 4.4 资产便携性

- 从下载几何重建六个带碰撞的 `*_physics.usd`；
- 在不覆盖原资产的前提下生成 `screwdriver_hybrid_physics.usd` 与 `wrench_sdf_physics.usd`；
- 修复原工具 USD 中失效的绝对 collider payload；
- 从 GLB/URDF 资产重建并规范化蓝箱；
- 所有运行时路径从安装包位置解析，USD 纹理使用相对路径；
- 删除包内未使用、含开发机导出元数据的 FBX 副本；
- 本地原始下载目录保持不动，但不进入正式提交归档。

### 4.5 工具链和真实验证

- 修复 stock teleop/record/replay 不会自动 import 外部 `bin_pickplace_tasks` 的注册问题；
- 新增真实 HDF5 动作重放、三视角拼接和 metric 验证视频导出器；
- 四目标 HDF5 到 MP4 的测试 probe 和四条真实人工演示均已完成端到端 CUDA 导出：官方形状的初始状态、7D 动作、`reset_to`、逐帧 metric、三相机、拼图与 H.264 编码均真实运行；probe 仍明确标记为非人工演示并只用于管线测试，最终提交视频来自真实人类动作；
- 15 个 CPU 单元测试全部通过，包括指标几何、重置布局、recorder 低显存 profile、目标顺序和 replay shape/目标恢复兼容层；
- RTX 5070 Ti 上混合碰撞配置的 240 步 CUDA smoke 通过：
  - action dimension = 7；
  - 三路 RGB 均为 `(1,480,640,3)`；
  - 三相机 frame counter 均为 241；
  - 8 秒内无隐式 reset；
  - 螺丝刀胶囊体/凸包与扳手 sparse SDF schema 检查通过并由 PhysX 成功加载；
  - 四件工具仍在白箱内、线速度均低于 0.001 m/s，最大 XY 漂移约 0.00036 m；
- CUDA live-scene 指标测试通过：
  - wrong tool、held、moving、rim 全部拒绝；
  - 正确稳定释放只在连续第 10 帧成功。
- CUDA recorder 顺序测试通过：成功计数 0..3 分别映射到 screwdriver、wrench、hammer、
  pallet knife，语言指令同步一致。
- 四工具 PhysX 实际质量核对通过：螺丝刀 `0.12 kg`、扳手 `0.25 kg`、锤子 `0.65 kg`、调色刀 `0.10 kg`；
- 以最重锤子等效的 `6.377 N` 重力和 `0.956 N·m` 偏心力矩测试机械臂保持：6 秒最大下沉从修复前 `198.096 mm` 降至 `5.032 mm`，最大姿态误差 `0.012475 rad`，控制目标漂移为 `0 mm`。
- CUDA 箱底接触实测发现：落稳螺丝刀旋转后 OBB 最低角点约为蓝箱局部 `z=0.00285 m`；指标
  已改为严格 `z>0`（实测外壳底面），避免旧 `z>0.020 m` 错误拒绝真正落在箱底的工具。

最新混合碰撞配置及落稳证据位于 `datasets/validation/hybrid_collision_final/`。

## 5. 完成状态、解释与交付边界

自动化工程部分已经完成并有真实 CUDA 证据。设计说明在 `REPORT.md`，运行入口和命令在
`README.md`。

四次真人键盘操作已经由提交者亲自完成，最终 HDF5 和四条 MP4 已保存在仓库中。四条视频不是脚本直接搬动物体得到的 probe；exporter 只重放记录的人类动作，并且只有再次触发目标感知的连续成功指标时才保留最终 MP4。

题面明确要求 `REPORT.md` 使用清晰英文，因此正式报告使用英文；本文件继续作为中文解释和交接说明。最终提交归档排除原始 `google_drive_assets/`、旧 build、egg-info、缓存和带保守 `valid:false` 的布局候选 JSON，只保留正式运行资产和最终证据。

提交前最后检查清单：

- [x] 外部项目可编辑安装；
- [x] 两个 Gym ID 可注册；
- [x] 场景、reset、7-D action、字典 observations；
- [x] 三相机 640x480 @ 30 Hz 同步；
- [x] target-aware、all-corner、released、settled、10-frame metric；
- [x] failure 和 timeout 分离；
- [x] portable runtime assets；
- [x] CPU 与 CUDA 自动测试；
- [x] README、REPORT、HOMEWORK_EXPLANATION 和 AI 使用披露；
- [x] 四条真人键盘 HDF5；
- [x] 四条 metric-complete 三视角 MP4；
- [x] 视频可完整解码、帧数/目标/指令/末帧指标检查；
- [ ] 提交者确认平台仍开放并上传最终归档。
