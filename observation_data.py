import torch
import numpy as np
from collections import deque


class ObservationData:
    def __init__(self, obs_buffer_size, data_buffer_size, batch_size):
        self.gpu_device = torch.device("cuda:0")
        self.cpu_device = torch.device("cpu:0")
        self.obs_buffer_size = obs_buffer_size
        self.data_buffer_size = data_buffer_size
        self.batch_size = batch_size
        self.obs_buffer = deque([], maxlen=self.obs_buffer_size)
        self.data_x = torch.zeros((self.data_buffer_size, self.obs_buffer_size, 3, 128, 128),
                                  dtype=torch.float32, device=self.cpu_device)
        self.data_y = torch.zeros((self.data_buffer_size, 3, 128, 128),
                                  dtype=torch.float32, device=self.cpu_device)
        self.counter = 0
        self.cur_top_down_obs = None
        self._reset_buffer()

    def _reset_buffer(self):
        for _ in range(self.obs_buffer_size):
            self.obs_buffer.append(torch.zeros((3, 128, 128), dtype=torch.float32, device=self.cpu_device))
        self.cur_top_down_obs = None

    def _process_obs(self, obs):
        return torch.tensor(obs, dtype=torch.float32, device=self.cpu_device).div_(255).permute(2, 0, 1)

    def _add_sample(self):
        data_ix = self.counter % self.data_buffer_size
        self.data_x[data_ix] = torch.stack(list(self.obs_buffer), 0)
        self.data_y[data_ix] = self._process_obs(self.cur_top_down_obs)
        self.counter += 1

    def append_obs(self, obs, top_down_obs, new_episode=False):
        if new_episode:
            self._reset_buffer()
        self.obs_buffer.append(self._process_obs(obs))
        if self.cur_top_down_obs is None:
            self.cur_top_down_obs = top_down_obs
        else:
            top_down_obs[top_down_obs == 0] = self.cur_top_down_obs[top_down_obs == 0]
            self.cur_top_down_obs = top_down_obs
        self._add_sample()

    def get_buffer(self):
        return torch.stack(list(self.obs_buffer), 0)

    def get_sample(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        available_data = min(self.counter, self.data_buffer_size)
        batch_ix = np.random.choice(available_data, batch_size)
        batch_x = self.data_x[batch_ix].to(self.gpu_device)
        batch_y = self.data_y[batch_ix].to(self.gpu_device)
        return batch_x, batch_y
