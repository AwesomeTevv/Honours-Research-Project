import setup_path
import airsim
import numpy as np
import math
from gym import spaces
from airgym.envs.airsim_env import AirSimEnv

class TestEnv(AirSimEnv):
    def __init__(self, ip_address, step_length, image_shape):
        super().__init__(image_shape)
        self.step_length = step_length
        self.image_shape = image_shape

        self.state = {
            "position": np.zeros(3),
            "collision": False,
            "prev_position": np.zeros(3),
        }

        self.drone = airsim.MultirotorClient(ip=ip_address)
        self.action_space = spaces.Discrete(7)
        self._setup_flight()

        self.lidar_request = airsim.LidarData()
        self.goal_position = np.array([50, 0, -10])  # 50 meters in front of starting position

    def __del__(self):
        self.drone.reset()

    def _setup_flight(self):
        self.drone.reset()
        self.drone.enableApiControl(True)
        self.drone.armDisarm(True)

        # Set home position and velocity
        self.drone.moveToPositionAsync(0, 0, -10, 5).join()
        self.drone.moveByVelocityAsync(0, 0, 0, 1).join()

    def _get_obs(self):
        self.drone_state = self.drone.getMultirotorState()
        self.state["prev_position"] = self.state["position"]
        self.state["position"] = self.drone_state.kinematics_estimated.position
        self.state["velocity"] = self.drone_state.kinematics_estimated.linear_velocity

        collision = self.drone.simGetCollisionInfo().has_collided
        self.state["collision"] = collision

        lidar_data = self.drone.getLidarData()
        if (len(lidar_data.point_cloud) < 3):
            print("No LiDAR data")
            return np.zeros(self.image_shape)

        points = np.array(lidar_data.point_cloud, dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0]/3), 3))
        
        lidar_image = np.zeros(self.image_shape[:2], dtype=np.uint8)
        for point in points:
            x = int((point[0] + 10) * 4)  # Assuming 20m range, 84x84 image
            y = int((point[1] + 10) * 4)
            if 0 <= x < self.image_shape[0] and 0 <= y < self.image_shape[1]:
                lidar_image[x, y] = 255

        return lidar_image.reshape(self.image_shape)

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
        dist_to_goal = np.linalg.norm(self.goal_position - np.array([
            self.state["position"].x_val,
            self.state["position"].y_val,
            self.state["position"].z_val
        ]))

        if self.state["collision"]:
            reward = -100
            done = True
        elif dist_to_goal < 3:  # Within 3 meters of the goal
            reward = 100
            done = True
        else:
            reward = -0.01 * dist_to_goal  # Small negative reward based on distance to goal
            done = False

        return reward, done

    def step(self, action):
        self._do_action(action)
        obs = self._get_obs()
        reward, done = self._compute_reward()

        return obs, reward, done, self.state

    def reset(self):
        self._setup_flight()
        return self._get_obs()

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