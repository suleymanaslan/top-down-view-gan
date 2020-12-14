import cv2
import torch
import numpy as np


class MultiViewData:
    def __init__(self, episode_duration, data_buffer_size, batch_size):
        self.gpu_device = torch.device("cuda:0")
        self.episode_duration = episode_duration
        self.data_buffer_size = data_buffer_size
        self.batch_size = batch_size
        self.data_x = np.zeros((self.data_buffer_size, 3, 2, 64, 64), dtype=np.uint8)
        self.data_y = np.zeros((self.data_buffer_size, 3, 64, 64), dtype=np.uint8)
        self.counter = 0
        self.cur_top_down_obs = None
        self._reset()

    def _reset(self):
        self.cur_top_down_obs = np.zeros((128, 128, 3), dtype=np.uint8)

    @staticmethod
    def _process_obs(obs):
        return cv2.resize(obs, (64, 64))

    def append_obs(self, obs, top_down_obs, new_episode=False):
        if new_episode:
            self._reset()
        data_ix = self.counter % self.data_buffer_size
        multiview_obs = np.concatenate((np.expand_dims(self._process_obs(obs), axis=0),
                                        np.expand_dims(self._process_obs(self.cur_top_down_obs), axis=0)),
                                       axis=0)
        self.data_x[data_ix] = multiview_obs.transpose((3, 0, 1, 2))
        if new_episode:
            self.cur_top_down_obs = top_down_obs
        else:
            top_down_obs[top_down_obs == 0] = self.cur_top_down_obs[top_down_obs == 0]
            self.cur_top_down_obs = top_down_obs
        self.data_y[data_ix] = self._process_obs(self.cur_top_down_obs).transpose((2, 0, 1))
        self.counter += 1

    def _to_torch(self, np_array):
        return (torch.from_numpy(np_array) / 255.0).to(self.gpu_device)

    def get_sample(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        available_data = min(self.counter, self.data_buffer_size)
        batch_ix = np.random.choice(available_data, batch_size)
        batch_x = self.data_x[batch_ix]
        batch_y = self.data_y[batch_ix]
        return self._to_torch(batch_x), self._to_torch(batch_y)

    def get_episode(self):
        episode_ix = np.random.choice(4760) + 1
        episode_x = self.data_x[episode_ix * self.episode_duration:(episode_ix + 1) * self.episode_duration]
        episode_y = self.data_y[episode_ix * self.episode_duration:(episode_ix + 1) * self.episode_duration]
        return self._to_torch(episode_x), self._to_torch(episode_y)

    def save(self, data_folder):
        np.save(f"{data_folder}/multiview/data_x.npy", self.data_x)
        np.save(f"{data_folder}/multiview/data_y.npy", self.data_y)

    def load(self, data_folder):
        self.data_x = np.load(f"{data_folder}/multiview/data_x.npy")
        self.data_y = np.load(f"{data_folder}/multiview/data_y.npy")
        self.counter = self.data_buffer_size
