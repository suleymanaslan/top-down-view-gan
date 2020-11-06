import torch
from collections import deque


class ObservationData:
    def __init__(self, buffer_size):
        self.device = torch.device("cuda:0")
        self.buffer_size = buffer_size
        self.buffer = deque([], maxlen=self.buffer_size)
        self._reset_buffer()

    def _reset_buffer(self):
        for _ in range(self.buffer_size):
            self.buffer.append(torch.zeros(3, 128, 128, device=self.device))

    def _process_observation(self, observation):
        return torch.tensor(observation, dtype=torch.float32, device=self.device).div_(255).permute(2, 0, 1)

    def _add_sample(self):
        pass

    def append_observation(self, observation, top_down_obs, new_episode=False):
        if new_episode:
            self._reset_buffer()
        self.buffer.append(self._process_observation(observation))
        self._add_sample()

    def get_buffer(self):
        return torch.stack(list(self.buffer), 0)
