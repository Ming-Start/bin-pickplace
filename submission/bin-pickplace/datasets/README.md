# 演示数据

本目录包含四条成功的人工键盘演示及其最终三相机视频。

| Episode | 目标工具 | HDF5 动作数 / MP4 帧数 | 视频时长 |
|---|---|---:|---:|
| `demo_0` | 螺丝刀（screwdriver） | 2,129 | 70.97 s |
| `demo_1` | 扳手（wrench） | 3,521 | 117.37 s |
| `demo_2` | 锤子（hammer） | 2,173 | 72.43 s |
| `demo_3` | 油漆刮刀（pallet knife） | 2,122 | 70.73 s |

`bin_pickplace_demos.hdf5` 为每条 episode 保存 7 维原始动作、初始状态、逐步场景状态、处理后动作、低维策略观测、目标元数据与标准语言指令。四条 episode 全部标记为成功。为避免长时间三相机录制触发 CUDA 显存耗尽，RGB 帧不缓存进 HDF5 文件。数据集属性 `camera_video_source` 记录了视频由录制的动作重新渲染这一事实。

`videos/demo_0.mp4` 至 `videos/demo_3.mp4` 由 `scripts/export_demo_videos.py` 完整重放每条人工动作序列生成。导出器在每个控制步同步采集外部相机 1、外部相机 2 与腕部相机的图像，叠加 episode 指令与指标状态，并且只有当重放再次触发环境成功指标时才生成最终视频。四条视频均为 H.264、yuv420p、1280x1024、30 fps，帧数与对应 HDF5 动作数完全一致。

`validation/hybrid_collision_final/` 包含最终混合碰撞配置的相机与重置质检图片。历史验证截图与布局编辑器候选文件被有意排除在干净提交归档之外。权威的初始位姿由环境配置与重置事件定义。

`scripts/generate_export_probe_dataset.py` 只是 HDF5 到视频流水线的测试工具。探针输出必须写到提交数据集之外，携带 `probe_only` 元数据，并被显著标注为 `NOT A HUMAN DEMONSTRATION`。它不能替代上述四条人工演示。
