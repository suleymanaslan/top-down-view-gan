import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from model import Model

model = Model(max_scale=4,
              steps_per_scale=int(25e3),
              batch_size=16,
              lr=1e-3)

stl10_data = datasets.STL10("datasets/stl10",
                            split="train+unlabeled",
                            folds=None,
                            transform=transforms.Compose([transforms.Resize(model.image_size),
                                                          transforms.CenterCrop(model.image_size),
                                                          transforms.ToTensor(),
                                                          ]),
                            download=True)
dataloader = torch.utils.data.DataLoader(stl10_data,
                                         batch_size=model.batch_size,
                                         drop_last=True,
                                         shuffle=True,
                                         pin_memory=True)

model.train(dataloader)
