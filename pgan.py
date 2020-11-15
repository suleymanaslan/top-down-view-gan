from model import Model
from observation_data import ObservationData

model = Model(max_scale=5,
              steps_per_scale=int(5e3),
              lr=1e-3)

obs_data = ObservationData(obs_buffer_size=21,
                           data_buffer_size=4096,
                           batch_size=16)
obs_data.load()

for step_i in range(int(50e3)):
    x, y = obs_data.get_sample()
    model.train_step(x, y)

model.save()
