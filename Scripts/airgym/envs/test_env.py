import setup_path
import airsim
import numpy as np
import math
import time
from gym import spaces
from airgym.envs.airsim_env import AirSimEnv

class TestEnv(AirSimEnv):
    def __init__(self, ip_address, step_length, image_shape):
        super().__init__(image_shape)
        self.step_length = step_length
        self.image_shape = image_shape

        self.observation_space = spaces.Dict({
            'depth_image': spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8),
            'distance_to_goal': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            'angle_to_goal': spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32),
            'velocity': spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            'acceleration': spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        })
        self.action_space = spaces.Box(low=np.array([-1, -1, -1]), high=np.array([1, 1, 1]), shape=(3,), dtype=np.float32)

        self.drone = airsim.MultirotorClient(ip=ip_address)
        self.drone.confirmConnection()
        self.drone.enableApiControl(True)
        self.drone.armDisarm(True)

        self.goal = np.array([21.7, -8.93, -1.63])

        self.image_request = airsim.ImageRequest(
            0, airsim.ImageType.DepthPerspective, True, False
        )

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
        im_final = np.array(image.resize((84, 84)).convert("L"))

        return im_final.reshape([84, 84, 1])
    
    def _get_obs(self):
        responses = self.drone.simGetImages([self.image_request])
        
        depth_image = self._transform_obs(responses)

        state = self.drone.getMultirotorState().kinematics_estimated

        position = self._get_position(state)
        velocity = self._get_velocity(state)
        acceleration = self._get_acceleration(state)

        distance_to_goal = self._get_distance_to_goal(position)
        angle_to_goal = self._get_angle_to_goal(state)

        obs = {
            'depth_image': depth_image,
            'distance_to_goal': distance_to_goal,
            'angle_to_goal': angle_to_goal,
            'velocity': velocity,
            'acceleration': acceleration,
        }

        return obs
    
    def _do_action(self, action):
        quad_offset = self.interpret_action(action)
        quad_vel = self.drone.getMultirotorState().kinematics_estimated.linear_velocity
        self.drone.moveByVelocityAsync(
            quad_vel.x_val + quad_offset[0],
            quad_vel.y_val + quad_offset[1],
            quad_vel.z_val + quad_offset[2],
            5,
        ).join()
    
    def _compute_reward(self):
        state = self.drone.getMultirotorState().kinematics_estimated

        position = self._get_position(state)
        distance_to_goal = self._get_distance_to_goal(position)

        velocity = self._get_velocity(state)

        # Get vector from drone to goal
        direction_to_goal = self.goal - position
        direction_to_goal_norm = direction_to_goal / np.linalg.norm(direction_to_goal)
        
        # Penalize movement away from goal
        direction_dot_product = np.dot(velocity, direction_to_goal_norm)

        done = False
        # reward = -500.0

        # if distance_to_goal < 1.0:
        #     reward = 100.0
        #     done = True
        # else:
        #     reward = -distance_to_goal
        #     done = False
        
        # if self._check_collision():
        #     reward -= 100
        #     done = True

            # Reward for moving towards the goal
        reward = direction_dot_product  # Positive if moving towards the goal
        
        # Add additional rewards and penalties (e.g., collision, proximity to goal)
        if distance_to_goal < 1.0:
            reward += 100  # Big reward for reaching the goal
            done = True
        if self._check_collision():
            reward -= 10  # Penalty for collision
            done = True
        
        return reward, done

    def reset(self):
        self._setup_flight()
        return self._get_obs()
    
    def close(self):
        self.__del__()
    
    def step(self, action):
        # self._do_action(action)
        vx, vy, vz = float(action[0]), float(action[1]), float(action[2])
        self.drone.moveByVelocityAsync(vx, vy, vz, duration=1).join()

        obs = self._get_obs()

        reward, done = self._compute_reward()

        info = {
            "velocity": obs["velocity"],
            "acceleration": obs["acceleration"],
            "distance_to_goal": obs["distance_to_goal"],
            "angle_to_goal": obs["angle_to_goal"],
            "depth_image": obs["depth_image"]
        }

        return obs, reward, done, info
    
    def interpret_action(self, action):
        if action == 0:
            quad_offset = (self.step_length, 0, 0)
        elif action == 1:
            quad_offset = (0, self.step_length, 0)
        elif action == 2:
            quad_offset = (0, 0, self.step_length)
        elif action == 3:
            quad_offset = (-self.step_length, 0, 0)
        elif action == 4:
            quad_offset = (0, -self.step_length, 0)
        elif action == 5:
            quad_offset = (0, 0, -self.step_length)
        else:
            quad_offset = (0, 0, 0)

        return quad_offset
    
    def render(self, mode='rgb_array'):
        if mode == 'rgb_array':
            responses = self.drone.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
            response = responses[0]
            img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
            img_rgb = img1d.reshape(response.height, response.width, 3)
            return img_rgb
        else:
            return np.array([])

    def _get_position(self, state: airsim.KinematicsState):
        return np.array([state.position.x_val, state.position.y_val, state.position.z_val])
    
    def _get_velocity(self, state: airsim.KinematicsState):
        return np.array([state.linear_velocity.x_val, state.linear_velocity.y_val, state.linear_velocity.z_val])
    
    def _get_acceleration(self, state: airsim.KinematicsState):
        return np.array([state.linear_acceleration.x_val, state.linear_acceleration.y_val, state.linear_acceleration.z_val])
    
    def _get_distance_to_goal(self, position: np.ndarray):
        return np.linalg.norm(position - self.goal)

    def _get_angle_to_goal(self, state: airsim.KinematicsState):
        position = self._get_position(state)
        return math.atan2(self.goal[1] - position[1], self.goal[0] - position[0]) - state.orientation.z_val
    
    def _check_collision(self):
        collision_info = self.drone.simGetCollisionInfo()
        return collision_info.has_collided