import cv2
import torch
import numpy as np
from collections import deque


class ObservationData:
    def __init__(self, obs_buffer_size, data_buffer_size, batch_size):
        self.gpu_device = torch.device("cuda:0")
        self.obs_buffer_size = obs_buffer_size
        self.data_buffer_size = data_buffer_size
        self.batch_size = batch_size
        self.obs_buffer = deque([], maxlen=self.obs_buffer_size)
        self.data_x = np.zeros((self.data_buffer_size, 3, self.obs_buffer_size, 64, 64), dtype=np.uint8)
        self.data_y = np.zeros((self.data_buffer_size, 3, 64, 64), dtype=np.uint8)
        self.counter = 0
        self.cur_top_down_obs = None
        self._reset_buffer()

    def _reset_buffer(self):
        for _ in range(self.obs_buffer_size):
            self.obs_buffer.append(np.zeros((3, 64, 64), dtype=np.uint8))
        self.cur_top_down_obs = None

    def _to_torch(self, np_array):
        return (torch.from_numpy(np_array) / 255.0).to(self.gpu_device)

    def _add_sample(self):
        data_ix = self.counter % self.data_buffer_size
        self.data_x[data_ix] = np.array(self.obs_buffer).transpose((1, 0, 2, 3))
        self.data_y[data_ix] = self._process_obs(self.cur_top_down_obs).transpose((2, 0, 1))
        self.counter += 1

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
        self._add_sample()

    def get_sample(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        available_data = min(self.counter, self.data_buffer_size)
        batch_ix = np.random.choice(available_data, batch_size)
        batch_x = self.data_x[batch_ix]
        batch_y = self.data_y[batch_ix]
        return self._to_torch(batch_x), self._to_torch(batch_y)

    def get_episode(self):
        episode_ix = np.random.choice(760) + 1
        episode_x = self.data_x[episode_ix * self.obs_buffer_size:(episode_ix + 1) * self.obs_buffer_size]
        episode_y = self.data_y[episode_ix * self.obs_buffer_size:(episode_ix + 1) * self.obs_buffer_size]
        return self._to_torch(episode_x), self._to_torch(episode_y)

    def save(self):
        np.save(f"../data/observation/data_x.npy", self.data_x)
        np.save(f"../data/observation/data_y.npy", self.data_y)

    def load(self):
        self.data_x = np.load(f"../data/observation/data_x.npy")
        self.data_y = np.load(f"../data/observation/data_y.npy")
        self.counter = self.data_buffer_size
