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
        self.max_duration = 45  # time limit
        self.max_distance = 15  # maximum distance
        
        self.goal_position = np.array([5.0, 0.0, -5.0])  # Specified goal position

        self._setup_flight()

    def _setup_flight(self):
        self.drone.reset()
        self.drone.enableApiControl(True)
        self.drone.armDisarm(True)

        # Set home position and velocity
        self.drone.moveToPositionAsync(0, 0, -5, 5).join()
        self.drone.moveByVelocityAsync(0, 0, 0, 1).join()

        self.start_position = self.drone.getMultirotorState().kinematics_estimated.position
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
        # Get the current drone position
        drone_state = self.drone.getMultirotorState().kinematics_estimated.position
        dist_to_goal = self.distance_to_goal(drone_state)
        
        # Get collision info
        collision_info = self.drone.simGetCollisionInfo()
        
        # Get velocity to check if the drone is moving
        velocity = self.drone.getMultirotorState().kinematics_estimated.linear_velocity
        
        # Initialize reward and done flag
        reward = 0
        done = False
        
        # Collision penalty
        if collision_info.has_collided:
            reward = -100
            print("I hit something...")
            done = True
        # Goal reached reward
        elif dist_to_goal < 1:  # If within 1 meter of the goal
            reward = 100
            print("I did it!")
            done = True
        # Out of bounds penalty
        elif self.distance_from_start(drone_state) > self.max_distance:
            reward = -75
            print("I strayed too far away.")
            done = True
        else:
            # Progress reward: Encourage getting closer to the goal
            previous_dist_to_goal = self.prev_dist_to_goal if hasattr(self, 'prev_dist_to_goal') else dist_to_goal
            progress = previous_dist_to_goal - dist_to_goal
            reward += 10 * progress
            
            # Time penalty: Less severe if closer to the goal
            time_elapsed = time.time() - self.start_time
            if time_elapsed > self.max_duration:
                # Scaled penalty based on distance to goal
                scaled_penalty = -5 * (dist_to_goal / self.max_distance)
                reward += scaled_penalty
                print("I took too long.")
                done = True
            else:
                # Regular time penalty
                reward -= 0.01
            
            # Penalize inactivity: Ensure the drone keeps moving
            if velocity.x_val**2 + velocity.y_val**2 + velocity.z_val**2 < 0.1**2:  # If speed is below a threshold
                reward -= 1
            
            # Store the current distance for the next step
            self.prev_dist_to_goal = dist_to_goal
        
        return reward, done

    def distance_to_goal(self, drone_state):
        drone_position = np.array([drone_state.x_val, drone_state.y_val, drone_state.z_val])
        return np.linalg.norm(self.goal_position - drone_position)

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
        print("Reset!")
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