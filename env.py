import gym
import math
import gym_miniworld
from env_utils import get_data
from observation_data import ObservationData


class Env:
    def __init__(self, obs_buffer_size, data_buffer_size, batch_size):
        self.env = gym.make("MiniWorld-PickupObjs-v0")
        self.env.domain_rand = True
        self.env.max_episode_steps = math.inf
        self.agent_fov = 90
        self.counter = 0

        self.obs_data = ObservationData(obs_buffer_size, data_buffer_size, batch_size)

    def step(self):
        if self.counter % 24 == 0:
            self.env.reset()
            self.env.agent.cam_fov_y = self.agent_fov
            obs, top_down_obs = get_data(self.env)
            self.obs_data.append_obs(obs, top_down_obs, new_episode=True)
        else:
            _, reward, done, info = self.env.step(self.env.actions.turn_right)
            obs, top_down_obs = get_data(self.env)
            self.obs_data.append_obs(obs, top_down_obs)
        return self.obs_data

    def close(self):
        self.env.close()
