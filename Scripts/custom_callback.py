import numpy as np
import matplotlib.pyplot as plt
import wandb
from stable_baselines3.common.callbacks import BaseCallback


class CustomCallback(BaseCallback):
    def __init__(
        self, save_freq: int, save_path: str, verbose: int = 0, model_name: str = ""
    ):
        super(CustomCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.model_name = model_name
        self.best_reward = -np.inf
        self.episode_counter = 0

        self.rewards = []
        self.velocities = []
        self.episode_lengths = []

        self.current_episode_length = 0

    def _on_step(self) -> bool:
        self.current_episode_length += 1

        if self.locals["dones"][0]:
            reward = np.sum(self.locals["rewards"])
            velocity = np.mean([info["velocity"] for info in self.locals["infos"]])
            distance_to_goal = self.locals["infos"][-1]["distance_to_goal"]
            angle_to_goal = self.locals["infos"][-1]["angle_to_goal"]

            episode_length = self.current_episode_length
            self.episode_lengths.append(episode_length)
            self.current_episode_length = 0

            # goal = self.locals["infos"][-1]["goal"]

            # lidar_data = self.locals["infos"][-1]["lidar_data"]
            # point_cloud = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)

            log_data = {
                "Episode Reward": reward,
                "Average Velocity": velocity,
                "Distance to Goal": distance_to_goal,
                "Angle to Goal": angle_to_goal,
                "Episode Length": episode_length,
                # "LiDAR Point Cloud": wandb.Object3D(point_cloud)
            }

            wandb.log(log_data)

            # Save the best model if reward improves
            if reward > self.best_reward:
                self.best_reward = reward
                self.model.save(
                    f"{self.save_path}/{self.model_name}_best_model_{self.episode_counter}"
                )
                print(
                    f"Best model saved with reward: {reward} at episode {self.episode_counter}"
                )

            # Save the model every `save_freq` episodes
            # if self.episode_counter % self.save_freq == 0:
            #     self.model.save(
            #         f"{self.save_path}/{self.model_name}_model_at_episode_{self.episode_counter}"
            #     )
            #     print(f"Model saved at episode {self.episode_counter}")

            self.episode_counter += 1

        return True
