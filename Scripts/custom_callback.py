import numpy as np
import wandb
from stable_baselines3.common.callbacks import BaseCallback
from io import BytesIO
import matplotlib.pyplot as plt


class CustomCallback(BaseCallback):
    def __init__(self, save_freq: int, save_path: str, verbose: int = 0):
        super(CustomCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.best_reward = -np.inf  # To track the best reward
        self.episode_counter = 0  # Track the number of episodes

        self.rewards = []
        self.velocities = []

    def _on_step(self) -> bool:
        if self.locals["dones"][0]:
            # Get current reward and velocity from the episode
            reward = np.sum(self.locals["rewards"])
            velocity = np.mean([info["velocity"] for info in self.locals["infos"]])
            lidar_mean_distance = self.locals["infos"][-1]["lidar_mean_distance"]
            lidar_density = self.locals["infos"][-1]["lidar_density"]
            lidar_variance = self.locals["infos"][-1]["lidar_variance"]
            
            self.rewards.append(reward)
            self.velocities.append(velocity)

            log_data = {
                "Episode Reward": reward,
                "Average Velocity": velocity,
                "LiDAR Mean Distance": lidar_mean_distance,
                "LiDAR Density": lidar_density,
                "LiDAR Variance": lidar_variance,
                "Distance to Goal": self.locals["infos"][-1]["distance_to_goal"],
                "Angle to Goal": self.locals["infos"][-1]["angle_to_goal"],
            }

            # Log all the data to Weights & Biases
            wandb.log(log_data)

            self.episode_counter += 1

            # Check if the current reward is the best and save the model
            if reward > self.best_reward:
                self.best_reward = reward
                self.model.save(f"{self.save_path}/lidar_best_model_{self.episode_counter}")
                print(f"Best model saved with reward: {reward} at episode {self.episode_counter}")

            # Save the model every `save_freq` episodes
            if self.episode_counter % self.save_freq == 0:
                self.model.save(f"{self.save_path}/lidar_model_at_episode_{self.episode_counter}")
                print(f"Model saved at episode {self.episode_counter}")
        
        return True