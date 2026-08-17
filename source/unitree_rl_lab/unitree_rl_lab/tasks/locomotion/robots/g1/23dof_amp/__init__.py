import gymnasium as gym

from . import agents


gym.register(
    id="Unitree-G1-23DoF-AMP-Walk-Direct-v0",
    entry_point=f"{__name__}.g1_amp_env:G1AmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_amp_env_cfg:G1AmpEnvCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_amp_cfg.yaml",
    },
)
