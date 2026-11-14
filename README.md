# Bin Pick-and-Place for Isaac Lab

This repository is a standalone, installable Isaac Lab manager-based environment for language-conditioned Franka Panda keyboard teleoperation. The operator must pick the instructed tool from a white source bin and release it inside a blue target bin. Two external RGB cameras and one wrist RGB camera observe the episode synchronously.

## Completion Status

The environment, portable assets, 7D relative differential-IK action, dictionary observations, reset logic, target-aware task metric, camera pipeline, teleoperation/record/replay wrappers, validation tools, human demonstrations, and final videos are complete.

| Episode | Target | Actions / video frames | Duration |
| --- | --- | ---: | ---: |
| `demo_0` | screwdriver | 2129 | 70.97 s |
| `demo_1` | wrench | 3521 | 117.37 s |
| `demo_2` | hammer | 2173 | 72.43 s |
| `demo_3` | pallet knife | 2122 | 70.73 s |

All final videos are H.264/yuv420p, 1280x1024, and 30 fps. Their frame counts exactly match the corresponding HDF5 action counts. Each action replay reached the environment's metric-detected success condition.

## Requirements

- Isaac Sim 5.1
- A compatible unmodified Isaac Lab checkout (validated with Isaac Lab 0.54.3)
- Python 3.11 in the activated Isaac Lab environment
- A CUDA-capable GPU for cameras, teleoperation, and video export

## Installation

From the repository root:

```bash
python -m pip install --no-build-isolation -e source/bin_pickplace_tasks
python scripts/validate_install.py
python scripts/list_envs.py
```

Registered environment IDs:

- `Bin-PickPlace-Franka-v0`: multi-environment configuration with random target sampling.
- `Bin-PickPlace-Franka-Play-v0`: deterministic one-environment evaluation configuration.

## Environment Contract

- Scene: Franka Panda, stock Seattle Lab table, white source bin, blue target bin, and four supplied tools.
- Instruction: `Pick the <tool_name> from the white bin and place it in the blue bin.`
- Action: `[dx, dy, dz, dRx, dRy, dRz, gripper]`.
- Arm control: relative 6-DoF damped-least-squares differential IK.
- Gripper control: one binary action.
- Physics: 120 Hz with control decimation 4.
- Control, rendering, observations, and cameras: synchronized at 30 Hz.
- Cameras: two external RGB views and one wrist RGB view, all 640x480.
- Success: the instructed target's eight local bounding-box corners are inside the blue-bin interior, the target is settled, the gripper is open and separated, and all conditions persist for ten control steps.
- Termination: success, unrecoverable target loss, and timeout are separate terms.

Detailed frames, thresholds, camera calibration, tradeoffs, validation, and AI disclosure are documented in `REPORT.md`.

## Zero-Action Validation

The required zero-agent entry point is:

```bash
python scripts/zero_agent.py \
  --task Bin-PickPlace-Franka-Play-v0 \
  --headless --enable_cameras --steps 300
```

## Keyboard Teleoperation

Set `ISAACLAB_ROOT` to the compatible Isaac Lab checkout:

```bash
export ISAACLAB_ROOT=/absolute/path/to/IsaacLab
python scripts/teleop.py \
  --task Bin-PickPlace-Franka-Play-v0 \
  --sensitivity 1.0
```

The wrapper preserves the active Python interpreter, imports `bin_pickplace_tasks` before launching the stock Isaac Lab tool, and automatically enables cameras.

At sensitivity `1.0`, the nominal translation command is approximately `0.012 m` per 30 Hz control step (`0.36 m/s`). A value near `2.0` is useful for free-space travel; `0.4-0.8` is recommended for final alignment and contact. The accepted range is `(0, 5]`.

Press `R` to reset. The deterministic environment prints the canonical instruction for the active target.

## Human Demonstration Recording

The final dataset is already included. To reproduce it, first preserve or move the existing file, then run:

```bash
export ISAACLAB_ROOT=/absolute/path/to/IsaacLab
python scripts/record_demos.py \
  --task Bin-PickPlace-Franka-Play-v0 \
  --teleop_device keyboard \
  --dataset_file datasets/bin_pickplace_demos.hdf5 \
  --num_demos 4 --num_success_steps 10
```

Successful episodes are assigned in canonical order:

```text
screwdriver -> wrench -> hammer -> pallet knife
```

The recording profile stores initial state, per-step scene states, raw and processed actions, and low-dimensional policy observations. It does not accumulate the three 480x640 RGB tensors in the episode buffer, which prevents long demonstrations from exhausting GPU memory. RGB is re-rendered from the recorded actions during final video export, and this deviation is recorded in the HDF5 metadata.

## Replay

Optional stock state/action replay:

```bash
export ISAACLAB_ROOT=/absolute/path/to/IsaacLab
python scripts/replay_demos.py \
  --task Bin-PickPlace-Franka-Play-v0 \
  --dataset_file datasets/bin_pickplace_demos.hdf5 \
  --validate_states
```

The compatibility wrapper removes only the synthetic leading singleton dimension introduced by the stock `EpisodeData.get_state()` path and restores target identity from each episode's HDF5 metadata after reset. The original elementwise `0.01` comparison tolerance is unchanged.

Do not use stock `--validate_success_rate` for this task. The stock tool calls the stateful persistent success term only once at episode end, which cannot validate a ten-frame condition. Target-aware per-frame success validation is performed by the exporter.

## Three-View Video Export

To regenerate the existing videos:

```bash
python scripts/export_demo_videos.py \
  --task Bin-PickPlace-Franka-Play-v0 \
  --dataset-file datasets/bin_pickplace_demos.hdf5 \
  --output-dir datasets/videos \
  --headless --enable_cameras --overwrite
```

The exporter restores each recorded initial state, replays only the human actions, renders the three synchronized views, adds camera labels and the language instruction, evaluates success every step, and refuses to commit a final MP4 unless the target-aware metric fires.

Final required files:

```text
datasets/videos/demo_0.mp4
datasets/videos/demo_1.mp4
datasets/videos/demo_2.mp4
datasets/videos/demo_3.mp4
```

`demo_1` contains approximately 24 seconds of initial idle time and completes 2.63 seconds before the 120-second timeout. It is valid but less concise than the other demonstrations.

## Validation

CPU tests:

```bash
python -m unittest -v \
  tests.test_metric_geometry \
  tests.test_recording_profile \
  tests.test_replay_profile
```

CUDA smoke test:

```bash
python scripts/smoke_test.py \
  --task Bin-PickPlace-Franka-Play-v0 \
  --headless --enable_cameras --steps 240 \
  --capture-dir datasets/validation/hybrid_collision_final
```

Live task-metric probe:

```bash
python scripts/probe_task_metrics.py \
  --task Bin-PickPlace-Franka-Play-v0 \
  --headless --enable_cameras
```

Mass and arm-holding probes:

```bash
python scripts/inspect_tool_masses.py --headless --enable_cameras

python scripts/probe_arm_load_holding.py \
  --headless --enable_cameras --steps 180 \
  --max-drop 0.006 --max-angle 0.02
```

## Portable Assets and Packaging

Runtime assets are stored under `source/bin_pickplace_tasks/bin_pickplace_tasks/assets` and resolved relative to the installed package. No runtime code or packaged asset depends on a development-machine home directory or the original download location.

The clean submission excludes raw downloads, stale build output, egg-info, caches, layout-editor candidate files, and historical validation snapshots. It retains the installable source, portable runtime assets, required scripts, tests, English documentation, HDF5 bonus dataset, four required MP4 videos, and the latest validation images.
