import gym
import airgym
import time
import wandb
from wandb.integration.sb3 import WandbCallback

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback,EvalCallback
from stable_baselines3.common.vec_env import VecVideoRecorder

# Initialize Weights & Biases
wandb.init(project="airsim-drone-rl")

# Custom callback for logging rewards
class RewardLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardLoggingCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Check if the episode is done
        if 'episode' in self.locals['infos'][0]:
            episode_reward = self.locals['infos'][0]['episode']['r']
            wandb.log({"episode_reward": episode_reward})
        return True

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
    learning_rate=0.0003,
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
    n_eval_episodes=1,
    best_model_save_path="./best_model",
    log_path="./logs",
    eval_freq=1,
)

# Create a Weights and Biases callback
wandb_callback = WandbCallback(
    gradient_save_freq=100,
    model_save_path=f"models/{int(time.time())}",
    verbose=2,
)

w_cb_2 = RewardLoggingCallback()

# Combine callbacks
callbacks = [eval_callback, wandb_callback, w_cb_2]

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