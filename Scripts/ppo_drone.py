import gym
import airgym
import time
import wandb
from wandb.integration.sb3 import WandbCallback

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import VecVideoRecorder

# Custom callback for logging episode rewards
class WandbEpisodeLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(WandbEpisodeLoggerCallback, self).__init__(verbose)
        self.episode_count = 0
        self.episode_reward = 0

    def _on_step(self) -> bool:
        # Accumulate reward
        self.episode_reward += self.locals["rewards"][0]

        # Check if episode has ended
        done = self.locals["dones"][0]
        if done:
            self.episode_count += 1
            # Log episode reward to wandb
            wandb.log({"episode": self.episode_count, "episode_reward": self.episode_reward})
            # Reset episode reward
            self.episode_reward = 0
        return True

# Initialize Weights & Biases
wandb.init(project="airsim-drone-rl")

# Create a DummyVecEnv for main airsim gym env
env = DummyVecEnv(
    [
        lambda: Monitor(
            gym.make(
                "airgym:airsim-v2",
                ip_address="127.0.0.1",
                step_length=0.25,
                image_shape=(84, 84, 2),
            )
        )
    ]
)

# Wrap env as VecTransposeImage to allow SB to handle frame observations
env = VecTransposeImage(env)

# # Add video recorder
# env = VecVideoRecorder(env, "videos", record_video_trigger=lambda x: x % 10000 == 0, video_length=200)

# Initialize PPO model
model = PPO(
    "CnnPolicy",
    env,
    verbose=1,
    tensorboard_log="./tb_logs/",
    learning_rate=0.001,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    device="cuda",
    policy_kwargs=dict(
        features_extractor_kwargs=dict(features_dim=256),
    ),
)

# Create an evaluation callback
eval_callback = EvalCallback(
    env,
    callback_on_new_best=None,
    n_eval_episodes=10,
    best_model_save_path="./best_model",
    log_path="./logs",
    eval_freq=10,
)

# Create a Weights and Biases callback
wandb_callback = WandbCallback(
    gradient_save_freq=100,
    model_save_path=f"models/{int(time.time())}",
    verbose=2,
)

# Create our custom episode logger callback
episode_logger_callback = WandbEpisodeLoggerCallback()

# Combine callbacks
callbacks = [eval_callback, wandb_callback, episode_logger_callback]

# Train the model
model.learn(
    total_timesteps=1_000,
    callback=callbacks,
    tb_log_name="ppo_airsim_drone_run_" + str(time.time()),
)

# Save the final model
model.save("ppo_airsim_drone_final_model")

# Close the environment
env.close()

# Finish the Weights & Biases run
wandb.finish()