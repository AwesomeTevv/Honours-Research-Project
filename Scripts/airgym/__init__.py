from gym.envs.registration import register

register(
    id="airsim-drone-v2", entry_point="airgym.envs:AirSimDroneEnv",
)
