# Demonstration Data

This directory contains four successful human keyboard demonstrations and their final three-camera videos.

| Episode | Target tool | HDF5 actions / MP4 frames | Video duration |
|---|---|---:|---:|
| `demo_0` | screwdriver | 2,129 | 70.97 s |
| `demo_1` | wrench | 3,521 | 117.37 s |
| `demo_2` | hammer | 2,173 | 72.43 s |
| `demo_3` | pallet knife | 2,122 | 70.73 s |

`bin_pickplace_demos.hdf5` stores 7D raw actions, the initial state, per-step scene states, processed actions, low-dimensional policy observations, target metadata, and the canonical language instruction for every episode. All four episodes are marked successful. To avoid CUDA out-of-memory failures during long three-camera recordings, RGB frames are not cached inside the HDF5 file. The dataset attribute `camera_video_source` records that the videos were re-rendered from the recorded actions.

`videos/demo_0.mp4` through `videos/demo_3.mp4` were produced by fully replaying each human action sequence with `scripts/export_demo_videos.py`. At every control step, the exporter captures synchronized images from External Camera 1, External Camera 2, and the wrist camera. It overlays the episode instruction and metric state, and it refuses to finalize a video unless the replay triggers the environment success metric again. All four videos are H.264, yuv420p, 1280x1024, and 30 fps. Each video frame count exactly matches its HDF5 action count.

`validation/hybrid_collision_final/` contains the final camera and reset QA images for the submitted hybrid collision configuration. Historical validation snapshots and layout-editor candidate files are intentionally excluded from the clean submission archive. The authoritative initial poses are defined by the environment configuration and reset event.

`scripts/generate_export_probe_dataset.py` is only a test utility for the HDF5-to-video pipeline. Probe output must be written outside the submission dataset, carries `probe_only` metadata, and is visibly marked `NOT A HUMAN DEMONSTRATION`. It cannot replace the four human demonstrations listed above.
