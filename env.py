import gym
import math
import torch
import cv2
import numpy as np
import gym_miniworld
from collections import deque
from env_utils import get_data


class Env:
    def __init__(self, obs_buffer_size):
        self.env = gym.make("MiniWorld-PickupObjs-v0")
        self.env.domain_rand = True
        self.env.max_episode_steps = math.inf
        self.agent_fov = 90
        self.counter = 0
        self.obs_buffer_size = obs_buffer_size
        self.obs_buffer = deque([], maxlen=self.obs_buffer_size)
        self.cur_top_down_obs = None
        self.device = torch.device("cuda:0")

    def _reset_buffer(self):
        for _ in range(self.obs_buffer_size):
            self.obs_buffer.append(np.zeros((3, 64, 64), dtype=np.uint8))
        self.cur_top_down_obs = None

    def _to_torch(self, np_array):
        return (torch.from_numpy(np_array) / 255.0).to(self.device)

    @staticmethod
    def _process_obs(obs):
        return cv2.resize(obs, (64, 64))

    def append_obs(self, obs, top_down_obs, new_episode=False):
        if new_episode:
            self._reset_buffer()
        self.obs_buffer.append(self._process_obs(obs).transpose((2, 0, 1)))
        if self.cur_top_down_obs is None:
            self.cur_top_down_obs = top_down_obs
        else:
            top_down_obs[top_down_obs == 0] = self.cur_top_down_obs[top_down_obs == 0]
            self.cur_top_down_obs = top_down_obs

    def step(self):
        if self.counter % self.obs_buffer_size == 0:
            self.env.reset()
            self.env.agent.cam_fov_y = self.agent_fov
            obs, top_down_obs = get_data(self.env)
            self.append_obs(obs, top_down_obs, new_episode=True)
        else:
            _, reward, done, info = self.env.step(self.env.actions.turn_right)
            obs, top_down_obs = get_data(self.env)
            self.append_obs(obs, top_down_obs)
        x = self._to_torch(np.array(self.obs_buffer).transpose((1, 0, 2, 3))).unsqueeze(0)
        y = self._to_torch(self._process_obs(self.cur_top_down_obs).transpose((2, 0, 1))).unsqueeze(0)
        self.counter += 1
        return x, y

    def close(self):
        self.env.close()
