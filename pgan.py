from model import Model
from env import Env

model = Model(max_scale=5,
              steps_per_scale=int(5e3),
              lr=1e-3)

env = Env(obs_buffer_size=25,
          data_buffer_size=1024,
          batch_size=16)

for step_i in range(int(50e3)):
    if step_i % 30 == 0:
        for _ in range(24):
            obs_data = env.step()
    x, y = obs_data.get_sample()
    model.train_step(x, y)

env.close()
