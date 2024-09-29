import airsim
import gym
import numpy as np
import airgym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.monitor import Monitor

import os
import sys


# Create a DummyVecEnv for main airsim gym env
env = DummyVecEnv(
    [
        lambda: Monitor(
            gym.make(
                "airgym:airsim-drone-cont-v1",
                ip_address="127.0.0.1",
                step_length=0.25,
                image_shape=(84, 84, 2),
            )
        )
    ]
)

# Wrap env as VecTransposeImage to allow SB to handle frame observations
# env = VecTransposeImage(env)

# Load the trained PPO model
model_loc = str(sys.argv[1])
print(f"Loading model from location: {model_loc}")

model = PPO.load(model_loc, weights_only=False)

obs = env.reset()

done = False
total_reward = 0

while not done:
    action, _states = model.predict(obs, deterministic=True)
    
    obs, reward, done, info = env.step(action)
    
    total_reward += reward
    
    print(f"Reward: {reward}, Total Reward: {total_reward}")

print(f"Episode finished with total reward: {total_reward}")

env.close()