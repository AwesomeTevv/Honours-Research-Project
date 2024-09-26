import gym
import numpy as np
import airgym
import time
import wandb

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

from custom_callback import CustomCallback

# Initialise Weights & Biases
wandb.init(project="airsim-drone-rl", sync_tensorboard=True)

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
env = VecTransposeImage(env)

# Initialise PPO model with custom policy
model = PPO(
    "MultiInputPolicy",
    env,
    verbose=1,
    tensorboard_log="../Logs/tb_logs/PPO/",
    device="cuda",
)

eval_callback = EvalCallback(
    env,
    callback_on_new_best=None,
    n_eval_episodes=5,
    best_model_save_path=f"../Models/PPO/SB/ppo_best_model_{int(time.time())}",
    log_path="../Logs/PPO/SB/",
    eval_freq=10,
)

custom_callback = CustomCallback()

# Combine callbacks
callbacks = [custom_callback]

# Train the model
model.learn(
    total_timesteps=1_000_000,
    callback=callbacks,
    tb_log_name="ppo_airsim_drone_run_" + str(time.time()),
)

# Save the final model
model.save("ppo_airsim_drone_final_model")

# Close the environment
env.close()

# Finish the Weights & Biases run
wandb.finish()