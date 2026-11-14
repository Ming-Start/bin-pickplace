"""Low-memory observation filtering for Bin-PickPlace demonstration recording."""

from __future__ import annotations

IMAGE_OBSERVATION_SUFFIXES = ("_rgb", "_depth", "_segmentation")
APP_LAUNCHER_ANCHOR = "simulation_app = app_launcher.app"
POST_APP_INSTALLATION = (
    "from recording_profile import install_low_memory_recording_profile\n"
    "install_low_memory_recording_profile()"
)


def inject_profile_after_app_launcher(source: str) -> str:
    """Inject the profile at a version-checked point in stock record_demos."""
    if source.count(APP_LAUNCHER_ANCHOR) != 1:
        raise RuntimeError(
            "Isaac Lab record_demos.py changed: expected exactly one AppLauncher anchor; "
            "refusing to run without the low-memory recorder patch"
        )
    return source.replace(
        APP_LAUNCHER_ANCHOR,
        APP_LAUNCHER_ANCHOR + "\n\n" + POST_APP_INSTALLATION,
        1,
    )


def filter_policy_observations(policy: dict) -> dict:
    """Return replay-useful low-dimensional terms without camera tensors."""
    if not isinstance(policy, dict):
        raise TypeError(
            "Bin-PickPlace low-memory recording requires concatenate_terms=False "
            "and a dictionary policy observation"
        )
    return {
        name: value
        for name, value in policy.items()
        if not name.endswith(IMAGE_OBSERVATION_SUFFIXES) and getattr(value, "ndim", 0) < 4
    }


def install_low_memory_recording_profile() -> None:
    """Patch the recorder after AppLauncher has initialized Isaac Sim/PXR."""
    from isaaclab.envs.mdp.recorders import recorders

    def record_state_only_policy_observations(self):
        return "obs", filter_policy_observations(self._env.obs_buf["policy"])

    recorders.PreStepFlatPolicyObservationsRecorder.record_pre_step = (
        record_state_only_policy_observations
    )
    print(
        "[Bin-PickPlace] low-memory recorder enabled: actions, scene states and "
        "low-dimensional observations are stored; RGB is re-rendered during video export.",
        flush=True,
    )
