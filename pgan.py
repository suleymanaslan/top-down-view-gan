from model import Model
from observation_data import ObservationData

model = Model(max_scale=4,
              steps_per_scale=int(25e3),
              lr=1e-3)

obs_data = ObservationData(obs_buffer_size=21,
                           data_buffer_size=int(16e3),
                           batch_size=16)
obs_data.load()

for step_i in range(int(400e3)):
    x, y = obs_data.get_sample()
    model.train_step(x, y)

model.save()
