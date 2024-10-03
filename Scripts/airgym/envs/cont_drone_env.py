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

        self.observation_space = spaces.Dict({
            'lidar_points': spaces.Box(low=-np.inf, high=np.inf, shape=(self.max_points, 3), dtype=np.float32),
            # 'lidar_mean_distance': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            # 'lidar_density': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            # 'lidar_variance': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            'distance_to_goal': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            'angle_to_goal': spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32),
            'velocity': spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
        })
        self.action_space = spaces.Box(low=np.array([-1, -1, -1]), high=np.array([1, 1, 1]), dtype=np.float32)

        self.drone = airsim.MultirotorClient(ip=ip_address)
        self.drone.confirmConnection()
        self.drone.enableApiControl(True)
        self.drone.armDisarm(True)

        self.goal = np.array([21.7, -8.93, -1.63])

        self.lidar_name = "LidarSensor1"

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
    
    def _process_lidar(self, lidar_points):
        if lidar_points.shape[0] == 0:
            return 0, 0, 0
        
        num_points = lidar_points.shape[0]
    
        lidar_points_fixed = np.zeros((self.max_points, 3), dtype=np.float32)
        lidar_points_fixed[:min(num_points, self.max_points), :] = lidar_points[:self.max_points, :]

        # distances = np.linalg.norm(lidar_points, axis=1)
        # mean_distance = np.mean(distances)

        # density = lidar_points.shape[0] / np.max(distances)

        # variance = np.var(distances)

        # return mean_distance, density, variance
        return lidar_points_fixed
    
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
        lidar_data = self.drone.getLidarData(lidar_name=self.lidar_name)

        if len(lidar_data.point_cloud) < 3:
            print("No LiDAR data available...")
            return self.reset()
        
        lidar_points = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)
        # lidar_mean, lidar_density, lidar_variance = self._process_lidar(lidar_points)
        lidar_points = self._process_lidar(lidar_points)

        state = self.drone.getMultirotorState().kinematics_estimated

        position = self._get_position(state)
        velocity = self._get_velocity(state)

        distance_to_goal = self._get_distance_to_goal(position)
        angle_to_goal = self._get_angle_to_goal(state)

        obs = {
            'lidar_points': lidar_points,
            # 'lidar_mean_distance': np.array([lidar_mean], dtype=np.float32),
            # 'lidar_density': np.array([lidar_density], dtype=np.float32),
            # 'lidar_variance': np.array([lidar_variance], dtype=np.float32),
            'distance_to_goal': distance_to_goal,
            'angle_to_goal': angle_to_goal,
            'velocity': velocity,
        }

        return obs, lidar_data, position
    
    def _compute_reward(self):
        state = self.drone.getMultirotorState().kinematics_estimated

        position = self._get_position(state)
        distance_to_goal = self._get_distance_to_goal(position)

        velocity = self._get_velocity(state)

        # Get vector from drone to goal
        direction_to_goal = self.goal - position
        direction_to_goal_norm = direction_to_goal / np.linalg.norm(direction_to_goal)
        
        # Penalise movement away from goal
        direction_dot_product = np.dot(velocity, direction_to_goal_norm)

        done = False

        # Reward for moving towards the goal
        reward = direction_dot_product  # Positive if moving towards the goal
        
        # Add additional rewards and penalties
        if distance_to_goal < 1.0:
            reward += 10
            print(f"Drone: {Format.GREEN}I made it!{Format.END} [{self.current_timestep}]", end=" ")
            done = True
        elif self.current_timestep >= self.max_timesteps:
            print(f"Drone: {Format.YELLOW}I took too long...{Format.END} [{self.current_timestep}]", end=" ")
            done = True
        
        if self._check_collision():
            reward -= 10  # Penalty for collision
            print(f"Drone: {Format.RED}I hit something...{Format.END} [{self.current_timestep}]", end=" ")
            done = True
        
        return reward, done

    def reset(self):
        self._setup_flight()
        self.current_timestep = 0
        return self._get_obs()[0]
    
    def close(self):
        self.__del__()
    
    def step(self, action):
        self.current_timestep += 1

        vx, vy, vz = float(action[0]), float(action[1]), float(action[2])
        # yaw_rate = float(action[3]) * 30 # Degrees per second

        # desired_yaw = math.atan2(vy, vx)
        # desired_yaw_degress = math.degrees(desired_yaw)

        # yaw_mode = airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate)
        # yaw_mode = airsim.YawMode(is_rate=False, yaw_or_rate=desired_yaw_degress)

        try:
            self.drone.moveByVelocityAsync(vx, vy, vz, duration=1).join()
            # self.drone.moveByVelocityAsync(vx, vy, vz, duration=2, yaw_mode=yaw_mode).join()
        except Exception as e:
            print(f"Error in moveByVelocityAsync: {e}")

        obs, lidar_points, position = self._get_obs()

        reward, done = self._compute_reward()
        if done:
            print(f"[{reward}]")

        info = {
            "velocity": obs["velocity"],
            "distance_to_goal": obs["distance_to_goal"],
            "angle_to_goal": obs["angle_to_goal"],
            # "lidar_mean_distance": obs["lidar_mean_distance"],
            # "lidar_density": obs["lidar_density"],
            # "lidar_variance": obs["lidar_variance"],
            "collision": self._check_collision(),
            "position": position,
            "lidar_data": lidar_points,
            "goal": self.goal
        }

        return obs, reward, done, info
    
    def render(self, mode='rgb_array'):
        return np.array([])

    def _get_position(self, state: airsim.KinematicsState):
        return np.array([state.position.x_val, state.position.y_val, state.position.z_val])
    
    def _get_velocity(self, state: airsim.KinematicsState):
        return np.array([state.linear_velocity.x_val, state.linear_velocity.y_val, state.linear_velocity.z_val])
    
    def _get_distance_to_goal(self, position: np.ndarray):
        return np.linalg.norm(position - self.goal)

    def _get_angle_to_goal(self, state: airsim.KinematicsState):
        position = self._get_position(state)
        return math.atan2(self.goal[1] - position[1], self.goal[0] - position[0]) - state.orientation.z_val
    
    def _check_collision(self):
        collision_info = self.drone.simGetCollisionInfo()
        return collision_info.has_collided