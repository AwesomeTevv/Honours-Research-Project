from gym.envs.registration import register

register(
    id="airsim-drone-sample-v0", entry_point="airgym.envs:AirSimDroneEnv",
)

register(
    id="airsim-v1", entry_point="airgym.envs:TestEnv",
)

register(
    id="airsim-v2", entry_point="airgym.envs:PPOEnv",
)