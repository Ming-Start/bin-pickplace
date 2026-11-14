import gymnasium as gym

gym.register(
    id="Bin-PickPlace-Franka-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.config.franka:FrankaBinPickPlaceEnvCfg"},
)

gym.register(
    id="Bin-PickPlace-Franka-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.config.franka:FrankaBinPickPlaceEnvCfg_PLAY"},
)

