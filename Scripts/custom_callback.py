import numpy as np
import matplotlib.pyplot as plt
import wandb
from stable_baselines3.common.callbacks import BaseCallback

class CustomCallback(BaseCallback):
    def __init__(self, save_freq: int, save_path: str, verbose: int = 0):
        super(CustomCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.best_reward = -np.inf  # To track the best reward
        self.episode_counter = 0  # Track the number of episodes

        self.rewards = []
        self.velocities = []
        self.episode_lengths = []
        self.collisions = []
        self.positions = []

        self.current_episode_length = 0

    def _on_step(self) -> bool:
        self.current_episode_length += 1

        if self.locals["dones"][0]:
            # Get current reward and velocity from the episode
            reward = np.sum(self.locals["rewards"])
            velocity = np.mean([info["velocity"] for info in self.locals["infos"]])
            distance_to_goal = self.locals["infos"][-1]["distance_to_goal"]
            angle_to_goal = self.locals["infos"][-1]["angle_to_goal"]
            lidar_mean_distance = self.locals["infos"][-1]["lidar_mean_distance"]
            lidar_density = self.locals["infos"][-1]["lidar_density"]
            lidar_variance = self.locals["infos"][-1]["lidar_variance"]
            collision_occurred = 1 if any(info.get("collision", False) for info in self.locals["infos"]) else 0
            # episode_length = len(self.locals["rewards"])

            self.rewards.append(reward)
            self.velocities.append(velocity)
            self.collisions.append(collision_occurred)

            episode_length = self.current_episode_length
            self.episode_lengths.append(episode_length)
            self.current_episode_length = 0

            positions = [info.get("position", [0, 0, 0]) for info in self.locals["infos"]]
            self.positions.append(positions)

            log_data = {
                "Episode Reward": reward,
                "Average Velocity": velocity,
                "Distance to Goal": distance_to_goal,
                "Angle to Goal": angle_to_goal,
                "LiDAR Mean Distance": lidar_mean_distance,
                "LiDAR Density": lidar_density,
                "LiDAR Variance": lidar_variance,
                "Episode Length": episode_length,
                "Collision Occurred": int(collision_occurred),
            }

            # Log all the data to Weights & Biases
            wandb.log(log_data)

            goal = self.locals["infos"][-1]["goal"]

            log_trajectory(positions, goal, step=self.num_timesteps)
            log_lidar_point_cloud(self.locals["infos"][-1]["lidar_data"], step=self.num_timesteps)
            log_lidar_feature_histogram(lidar_mean_distance, lidar_density, lidar_variance, step=self.num_timesteps)
            log_custom_reward_vs_goal_plot(self.rewards, [info["distance_to_goal"] for info in self.locals["infos"]], step=self.num_timesteps)
            self.episode_counter += 1

            # Check if the current reward is the best and save the model
            if reward > self.best_reward:
                self.best_reward = reward
                self.model.save(f"{self.save_path}/lidar2_best_model_{self.episode_counter}")
                print(f"Best model saved with reward: {reward} at episode {self.episode_counter}")

            # Save the model every `save_freq` episodes
            if self.episode_counter % self.save_freq == 0:
                self.model.save(f"{self.save_path}/lidar2_model_at_episode_{self.episode_counter}")
                print(f"Model saved at episode {self.episode_counter}")
        
        return True


def log_trajectory(positions, goal, step):
    """Logs the drone's trajectory as a 2D path in WandB."""
    traj_table = wandb.Table(columns=["x", "y", "z", "type"])
    
    # Log all the positions of the drone
    for pos in positions:
        traj_table.add_data(pos[0], pos[1], pos[2], "drone")
    
    # Log the goal position
    traj_table.add_data(goal[0], goal[1], goal[2], "goal")
    
    # Log the trajectory as a 3D plot in WandB
    wandb.log({"Drone Trajectory": traj_table}, step=step)

def log_lidar_point_cloud(lidar_data, step):
    """Logs LiDAR point cloud data as a 3D scatter plot in WandB."""
    # Extract the point cloud and reshape it into (N, 3) array (x, y, z for each point)
    point_cloud = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)
    
    # Log the NumPy array directly as a 3D object in WandB
    wandb.log({"LiDAR Point Cloud": wandb.Object3D(point_cloud)}, step=step)

def log_reward_heatmap(rewards, step):
    """Logs a heatmap of the rewards during the episode."""
    rewards_array = np.array(rewards).reshape(1, -1)  # Reshape to 1xN for heatmap
    
    # Log the heatmap
    wandb.log({
        "Reward Heatmap": wandb.Image(wandb.plots.HeatMap(
            np.arange(len(rewards)),
            ["Episode Rewards"],
            rewards_array
        ))
    }, step=step)

def log_lidar_feature_histogram(lidar_mean_distance, lidar_density, lidar_variance, step):
    """Logs histograms of LiDAR features like mean distance, density, and variance."""
    wandb.log({
        "LiDAR Mean Distance": wandb.Histogram(lidar_mean_distance),
        "LiDAR Density": wandb.Histogram(lidar_density),
        "LiDAR Variance": wandb.Histogram(lidar_variance)
    }, step=step)

def log_custom_reward_vs_goal_plot(rewards, distance_to_goals, step):
    """Creates a 2D plot of rewards vs. distance to goal and logs it to WandB."""
    fig, ax = plt.subplots()
    ax.plot(rewards, label="Rewards", color='b')
    ax.plot(distance_to_goals, label="Distance to Goal", color='r')
    
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Values")
    ax.set_title("Rewards and Distance to Goal Over Time")
    ax.legend()

    # Log the plot as an image in WandB
    wandb.log({"Rewards vs. Distance Plot": wandb.Image(fig)}, step=step)
    plt.close(fig)