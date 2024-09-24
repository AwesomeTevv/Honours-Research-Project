import setup_path
import gym
import airgym
import time
import wandb

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from wandb.integration.sb3 import WandbCallback

class WandbEpisodeLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(WandbEpisodeLoggerCallback, self).__init__(verbose)
        self.episode_reward = 0

    def _on_step(self) -> bool:
        self.episode_reward += self.locals["rewards"][0]

        done = self.locals["dones"][0]
        if done:
            wandb.log({"episode_reward": self.episode_reward})
            self.episode_reward = 0
        return True

wandb.init(project="airsim-drone-rl", sync_tensorboard=True)

# Create a DummyVecEnv for main airsim gym env
env = DummyVecEnv(
    [
        lambda: Monitor(
            gym.make(
                "airgym:airsim-drone-disc-v1",
                ip_address="127.0.0.1",
                step_length=0.25,
                image_shape=(84, 84, 1),
            )
        )
    ]
)

# Wrap env as VecTransposeImage to allow SB to handle frame observations
env = VecTransposeImage(env)

# Initialize RL algorithm type and parameters
model = DQN(
    "CnnPolicy",
    env,
    learning_rate=0.00025,
    verbose=1,
    batch_size=32,
    train_freq=4,
    target_update_interval=10000,
    learning_starts=10000,
    buffer_size=500000,
    max_grad_norm=10,
    exploration_fraction=0.1,
    exploration_final_eps=0.01,
    device="cuda",
    tensorboard_log="./logs/tb_logs/",
)

# Create a Weights and Biases callback
wandb_callback = WandbCallback(
    gradient_save_freq=100,
    model_save_path=f"models/{int(time.time())}",
    verbose=2,
)

# Create our custom episode logger callback
episode_logger_callback = WandbEpisodeLoggerCallback()

# Create an evaluation callback with the same env, called every 10000 iterations
callbacks = []
eval_callback = EvalCallback(
    env,
    callback_on_new_best=None,
    n_eval_episodes=5,
    best_model_save_path="./models/dqn_best_model",
    log_path="./logs/dqn",
    eval_freq=10000,
)

# Combine callbacks
callbacks = [eval_callback, wandb_callback, episode_logger_callback]

kwargs = {}
kwargs["callback"] = callbacks

# Train for a certain number of timesteps
model.learn(
    total_timesteps=5e5,
    tb_log_name="dqn_airsim_drone_run_" + str(time.time()),
    **kwargs
)

# Save policy weights
model.save("dqn_airsim_drone_policy")
