from model import Model
from multiview_data import MultiViewData

model = Model(max_scale=4,
              steps_per_scale=int(25e3),
              lr=1e-3,
              multiview=True)

multiview_data = MultiViewData(episode_duration=21,
                               data_buffer_size=int(1e5),
                               batch_size=16)
multiview_data.load()

for step_i in range(int(400e3)):
    x, y = multiview_data.get_sample()
    model.train_step(x, y)

model.save()
