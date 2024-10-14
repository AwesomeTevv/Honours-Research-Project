import gym
import airgym
import time
import wandb
import sys

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from custom_callback import CustomCallback

# Initialise Weights & Biases
wandb.init(project="airsim-drone-rl")

model_name = ""
if len(sys.argv) > 2:
    model_name = sys.argv[2]

# Create a DummyVecEnv for main airsim gym env
env = DummyVecEnv(
    [
        lambda: Monitor(
            gym.make(
                "airgym:airsim-drone-cont-v1",
                ip_address="127.0.0.1",
                step_length=0.25,
                image_shape=(84, 84, 1),
            )
        )
    ]
)

# Initialise PPO model with custom policy
model = PPO(
    "MultiInputPolicy",
    env,
    verbose=1,
    n_steps=200,
    batch_size=50,
    learning_rate=0.001,
    tensorboard_log="../Logs/PPO/TB/",
    device="cuda",
)

custom_callback = CustomCallback(
    save_freq=5, save_path="../Models/PPO/SB/", model_name=model_name
)

# Combine callbacks
callbacks = [custom_callback]

# num_epochs = 1000  # Total number of epochs
# num_episodes = 1  # Number of episodes per epoch
# num_timesteps = 100  # Number of timesteps per episode

# total_timesteps = num_epochs * num_episodes * num_timesteps

model.policy_kwargs = {
    "ent_coef": 0.01,  # Encouraging early exploration
}

# Train the model
model.learn(
    total_timesteps=100_000,
    callback=callbacks,
    tb_log_name="ppo_airsim_drone_run_" + str(time.time()),
)

# Save the final model
model.save(f"ppo_airsim_drone_final_model_{model_name}")

# Close the environment
env.close()

# Finish the Weights & Biases run
wandb.finish()
