import setup_path
import airsim
import numpy as np
import math
import time
from gym import spaces
from airgym.envs.airsim_env import AirSimEnv

class PPOEnv(AirSimEnv):
    def __init__(self, ip_address, step_length, image_shape):
        super().__init__(image_shape)
        self.step_length = step_length
        self.image_shape = image_shape

        self.drone = airsim.MultirotorClient(ip=ip_address)
        self.drone.confirmConnection()

        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 2), dtype=np.uint8)

        self.start_time = None
        self.max_duration = 30  # 30 seconds time limit
        self.max_distance = 15  # 15 meters maximum distance
        self.target_distance = 5  # 5 meters ahead

        self._setup_flight()

    def _setup_flight(self):
        self.drone.reset()
        self.drone.enableApiControl(True)
        self.drone.armDisarm(True)

        self.start_position = self.drone.getMultirotorState().kinematics_estimated.position
        self.target_position = airsim.Vector3r(
            self.start_position.x_val + self.target_distance,
            self.start_position.y_val,
            self.start_position.z_val
        )
        self.drone.moveToPositionAsync(self.start_position.x_val, self.start_position.y_val, self.start_position.z_val, 5).join()
        self.start_time = time.time()

    def _get_obs(self):
        responses = self.drone.simGetImages([airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True, False)])
        depth_img = self.transform_obs(responses)
        
        lidar_data = self.drone.getLidarData()
        lidar_points = np.array(lidar_data.point_cloud, dtype=np.dtype('f4'))
        lidar_points = np.reshape(lidar_points, (int(lidar_points.shape[0]/3), 3))
        
        # Simplify LiDAR data to a 2D grid
        lidar_2d = np.zeros((84, 84))
        for point in lidar_points:
            x, y = int((point[0] + 10) * 4.2), int((point[1] + 10) * 4.2)
            if 0 <= x < 84 and 0 <= y < 84:
                lidar_2d[x, y] = 255  # Set to 255 for visibility
        
        # Combine depth image and LiDAR data
        obs = np.dstack((depth_img.squeeze(), lidar_2d))
        return obs.astype(np.uint8)

    def _do_action(self, action):
        quad_offset = self.interpret_action(action)
        quad_vel = self.drone.getMultirotorState().kinematics_estimated.linear_velocity
        self.drone.moveByVelocityAsync(
            quad_vel.x_val + quad_offset[0],
            quad_vel.y_val + quad_offset[1],
            quad_vel.z_val + quad_offset[2],
            5
        ).join()

    def _compute_reward(self):
        drone_state = self.drone.getMultirotorState().kinematics_estimated.position
        dist_to_target = self.distance_to_target(drone_state)
        
        collision_info = self.drone.simGetCollisionInfo()
        
        if collision_info.has_collided:
            reward = -100
            done = True
        elif dist_to_target < 1:  # If within 1 meter of the target
            reward = 100
            done = True
        elif self.distance_from_start(drone_state) > self.max_distance:
            reward = -50
            done = True
        elif time.time() - self.start_time > self.max_duration:
            reward = -50
            done = True
        else:
            reward = -dist_to_target  # Negative distance as reward
            done = False
        
        return reward, done

    def distance_to_target(self, drone_state):
        return np.linalg.norm(
            np.array([self.target_position.x_val, self.target_position.y_val, self.target_position.z_val]) - 
            np.array([drone_state.x_val, drone_state.y_val, drone_state.z_val])
        )

    def distance_from_start(self, drone_state):
        return np.linalg.norm(
            np.array([self.start_position.x_val, self.start_position.y_val, self.start_position.z_val]) - 
            np.array([drone_state.x_val, drone_state.y_val, drone_state.z_val])
        )

    def step(self, action):
        self._do_action(action)
        obs = self._get_obs()
        reward, done = self._compute_reward()
        info = {}
        return obs, reward, done, info

    def reset(self):
        self._setup_flight()
        return self._get_obs()

    def interpret_action(self, action):
        scaled_action = np.clip(action, -1, 1) * self.step_length
        return scaled_action

    def transform_obs(self, responses):
        img1d = np.array(responses[0].image_data_float, dtype=np.float)
        img1d = 255 / np.maximum(np.ones(img1d.size), img1d)
        img2d = np.reshape(img1d, (responses[0].height, responses[0].width))

        from PIL import Image
        image = Image.fromarray(img2d)
        im_final = np.array(image.resize((84, 84)).convert("L"))

        return im_final.reshape([84, 84, 1])
    
    def render(self, mode='rgb_array'):
        if mode == 'rgb_array':
            responses = self.drone.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
            response = responses[0]
            img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
            img_rgb = img1d.reshape(response.height, response.width, 3)
            return img_rgb
        else:
            return np.array([])