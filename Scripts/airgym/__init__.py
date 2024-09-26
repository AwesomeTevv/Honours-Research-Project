from gym.envs.registration import register

register(
    id="airsim-drone-cont-v1", entry_point="airgym.envs:ContDroneEnv",
)

register(
    id="airsim-drone-disc-v1", entry_point="airgym.envs:DiscDroneEnv",
)