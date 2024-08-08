import setup_path
import airsim
import numpy as np
import math
import time
from argparse import ArgumentParser

import gym
from gym import spaces
from airgym.envs.airsim_env import AirSimEnv

class AirSimDroneEnv(AirSimEnv):
    def __init__(self, ip_address, step_length, image_shape):
        super().__init__(image_shape)
        self.step_length = step_length
        self.image_shape = image_shape

        self.state = {
            "position": np.zeros(3),
            "collision": False,
            "prev_position": np.zeros(3)
        }

        self.goal = np.array([50, 50, -10]) # Goal position

        self.drone = airsim.MultirotorClient(ip=ip_address)
        self.action_space = spaces.Discrete(7)
        self._setup_flight()

        self.image_request = airsim.ImageRequest(
            3, airsim.ImageType.DepthPerspective, True, False
        )
    
    def __del__(self):
        self.drone.reset()
    
    def _setup_flight(self):
        self.drone.reset()
        self.drone.enableApiControl(True)
        self.drone.armDisarm(True)

        # Set home position and velocity
        self.drone.moveToPositionAsync(0, 0, -10, 50).join()
        self.drone.moveByVelocityAsync(0, 0, 0, 5).join()
    
    def transform_obs(self, responses):
        img1D = np.array(responses[0].image_data_float, dtype=np.float32)
        img1D = 255 / np.maximum(np.ones(img1D.size), img1D)
        
        img2D = np.reshape(img1D, (responses[0].height, responses[0].width))

        from PIL import Image

        image = Image.fromarray(img2D)

        img_final = np.array(image.resize((84, 84)).convert("L"))

        return img_final.reshape([84, 84, 1])
    
    def _get_obs(self):
        responses = self.drone.simGetImages([self.image_request])
        image = self.transform_obs(responses)
        self.drone_state = self.drone.getMultirotorState()

        self.state["prev_position"] = self.state["position"]
        self.state["position"] = self.drone_state.kinematics_estimated.position
        self.state["velocity"] = self.drone_state.kinematics_estimated.linear_velocity

        collision = self.drone.simGetCollisionInfo().has_collided
        self.state["collision"] = collision

        return image
    
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
        current_position = self.state["position"]
        goal_position = self.goal

        distance = np.linalg.norm([
            current_position.x_val - goal_position[0],
            current_position.y_val - goal_position[1],
            current_position.z_val - goal_position[2],
        ])

        if self.state["collision"]:
            reward = -100
        else:
            reward = -distance
        
        done = 0
        if distance < 1: # Consider the episode done if the drone is very close to the goal
            done = 1
        
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


    
