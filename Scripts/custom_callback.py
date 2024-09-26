import numpy as np
import wandb
from stable_baselines3.common.callbacks import BaseCallback

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