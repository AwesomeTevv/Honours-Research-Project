from gym.envs.registration import register

register(
    id="airsim-drone-cont-v1", entry_point="airgym.envs:ContAirSimDroneEnv",
)

register(
    id="airsim-drone-disc-v1", entry_point="airgym.envs:DiscAirSimDroneEnv",
)
