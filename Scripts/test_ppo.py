import gym
import numpy as np
import airgym
import time
import wandb
from wandb.integration.sb3 import WandbCallback

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

class CustomCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super(CustomCallback, self).__init__(verbose)

        self.rewards = []
        self.velocities = []
        self.accelerations = []
    
    def _on_step(self) -> bool:
        if self.locals["dones"][0]:
            reward = np.sum(self.locals["rewards"])
            velocity = np.mean([info["velocity"] for info in self.locals["infos"]])
            acceleration = np.mean([info["acceleration"] for info in self.locals["infos"]])
            # depth_image = self.locals["infos"][-1]["depth_image"]

            self.rewards.append(reward)
            self.velocities.append(velocity)
            self.accelerations.append(acceleration)

            wandb.log({
                "Episode Reward": reward,
                "Average Velocity": velocity,
                "Average Acceleration": acceleration,
                "Distance to Goal": self.locals["infos"][-1]["distance_to_goal"],
                "Angle to Goal": self.locals["infos"][-1]["angle_to_goal"],
                # "Depth Image": wandb.Image(depth_image, caption="Depth Image")
            })
        
        return True

# Initialise Weights & Biases
wandb.init(project="airsim-drone-rl", sync_tensorboard=True)

# Create a DummyVecEnv for main airsim gym env
env = DummyVecEnv(
    [
        lambda: Monitor(
            gym.make(
                "airgym:test-v0",
                ip_address="127.0.0.1",
                step_length=0.25,
                image_shape=(84, 84, 2),
            )
        )
    ]
)

# Initialise PPO model with custom policy
model = PPO(
    "MultiInputPolicy",
    env,
    verbose=1,
    tensorboard_log="../Logs/tb_logs/PPO/",
    device="cuda",
)

# eval_callback = EvalCallback(
#     env,
#     callback_on_new_best=None,
#     n_eval_episodes=10,
#     best_model_save_path=f"../Models/PPO/SB/ppo_best_model_{int(time.time())}",
#     log_path="../Logs/PPO/SB/",
#     eval_freq=10,
# )

# Create a Weights and Biases callback
wandb_callback = WandbCallback(
    gradient_save_freq=10,
    model_save_path=f"../Models/PPO/WandB/{int(time.time())}",
    verbose=2,
)

custom_callback = CustomCallback()

# Combine callbacks
callbacks = [wandb_callback, custom_callback]

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