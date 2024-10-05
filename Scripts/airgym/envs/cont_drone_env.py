import setup_path
import airsim
import numpy as np
import math
from gym import spaces
from airgym.envs.airsim_env import AirSimEnv

from style import Format


class ContDroneEnv(AirSimEnv):
    def __init__(self, ip_address, step_length, image_shape):
        super().__init__(image_shape)
        self.step_length = step_length
        self.image_shape = image_shape

        self.max_timesteps = 200
        self.current_timestep = 0

        self.max_points = 10_000

        self.observation_space = spaces.Dict(
            {
                "distance_to_goal": spaces.Box(
                    low=0, high=np.inf, shape=(1,), dtype=np.float32
                ),
                "angle_to_goal": spaces.Box(
                    low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32
                ),
                "velocity": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
                ),
                "position": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
                ),
                "depth_image": spaces.Box(
                    low=0, high=255, shape=self.image_shape, dtype=np.float32
                ),
            }
        )
        self.action_space = spaces.Box(
            low=np.array([-1, -1, -1]),
            high=np.array([1, 1, 1]),
            dtype=np.float32,
        )

        self.drone = airsim.MultirotorClient(ip=ip_address)
        self.drone.confirmConnection()
        self.drone.enableApiControl(True)
        self.drone.armDisarm(True)

        # self.goal = np.array([21.7, -8.93, -1.63])
        # self.goal = np.array([3.81, -60.82, 12.36])
        self.goal = np.array([-29.46, -43.01, 0.00])

        # self.lidar_name = "LidarSensor1"
        self.sensor_name = "Distance"

        self._setup_flight()

    def __del__(self):
        self.drone.reset()

    def _setup_flight(self):
        self.drone.reset()
        self.drone.enableApiControl(True)
        self.drone.armDisarm(True)

        # Set home position and velocity
        self.drone.moveToPositionAsync(0, 0, -1, 5).join()
        self.drone.moveByVelocityAsync(0, 0, 0, 1).join()

    def _transform_obs(self, responses):
        response = responses[0]

        img1d = np.array(response.image_data_float, dtype=np.float32)
        img1d = 255 / np.maximum(np.ones(img1d.size), img1d)
        img2d = np.reshape(img1d, (response.height, response.width))

        from PIL import Image

        image = Image.fromarray(img2d)
        im_final = np.array(image.resize((256, 256)).convert("L"))

        return im_final.reshape([256, 256, 1])

    def _get_obs(self):
        depth_data = self._transform_obs(
            self.drone.simGetImages(
                [airsim.ImageRequest(0, airsim.ImageType.DepthPerspective, True, False)]
            )
        )

        state = self.drone.getMultirotorState().kinematics_estimated

        position = self._get_position(state)
        velocity = self._get_velocity(state)

        distance_to_goal = self._get_distance_to_goal(position)
        angle_to_goal = self._get_angle_to_goal(state)

        obs = {
            "depth_image": depth_data,
            "distance_to_goal": distance_to_goal,
            "angle_to_goal": angle_to_goal,
            "velocity": velocity,
            "position": position,
        }

        return obs

    def _compute_reward(self):
        state = self.drone.getMultirotorState().kinematics_estimated

        position = self._get_position(state)
        distance_to_goal = self._get_distance_to_goal(position)

        velocity = self._get_velocity(state)

        direction_to_goal = self.goal - position
        direction_to_goal_norm = direction_to_goal / np.linalg.norm(direction_to_goal)

        direction_dot_product = np.dot(velocity, direction_to_goal_norm)

        done = False

        reward = direction_dot_product  # Positive if moving towards the goal

        if distance_to_goal < 1.0:
            reward += 10
            print(
                f"Drone: {Format.GREEN}Made it!{Format.END}\t[t={self.current_timestep}]",
                end="\t",
            )
            done = True
        elif self.current_timestep >= self.max_timesteps:
            print(
                f"Drone: {Format.YELLOW}Too long{Format.END}\t[t={self.current_timestep}]",
                end="\t",
            )
            done = True

        if self._check_collision():
            reward -= 10  # Penalty for collision
            print(
                f"Drone: {Format.RED}Collided{Format.END}\t[t={self.current_timestep}]",
                end="\t",
            )
            done = True

        return reward, done

    def reset(self):
        self._setup_flight()
        self.current_timestep = 0
        return self._get_obs()

    def close(self):
        self.__del__()

    def step(self, action):
        self.current_timestep += 1

        vx, vy, vz = float(action[0]), float(action[1]), float(action[2])

        try:
            self.drone.moveByVelocityAsync(vx, vy, vz, duration=1).join()
        except Exception as e:
            print(f"Error in moveByVelocityAsync: {e}")

        obs = self._get_obs()

        reward, done = self._compute_reward()

        if done:
            print(f"[r={reward:.2f}]")

        info = {
            "velocity": obs["velocity"],
            "distance_to_goal": obs["distance_to_goal"],
            "angle_to_goal": obs["angle_to_goal"],
            "position": obs["position"],
            "depth_image": obs["depth_image"],
            "goal": self.goal,
        }

        return obs, reward, done, info

    def render(self, mode="rgb_array"):
        return np.array([])

    def _get_position(self, state: airsim.KinematicsState):
        return np.array(
            [state.position.x_val, state.position.y_val, state.position.z_val]
        )

    def _get_velocity(self, state: airsim.KinematicsState):
        return np.array(
            [
                state.linear_velocity.x_val,
                state.linear_velocity.y_val,
                state.linear_velocity.z_val,
            ]
        )

    def _get_distance_to_goal(self, position: np.ndarray):
        return np.linalg.norm(position - self.goal)

    def _get_angle_to_goal(self, state: airsim.KinematicsState):
        position = self._get_position(state)
        return (
            math.atan2(self.goal[1] - position[1], self.goal[0] - position[0])
            - state.orientation.z_val
        )

    def _check_collision(self):
        collision_info = self.drone.simGetCollisionInfo()
        return collision_info.has_collided
