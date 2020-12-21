from skimage.measure import compare_ssim
import numpy as np
import cv2
from model.model import Model
from dataset_utils.observation_data import ObservationData
from dataset_utils.multiview_data import MultiViewData
from environment.env import Env


def get_results(train=True):
    psnr = []
    ssim = []
    for j in range(10):
        episode_x, y = data.get_episode() if train else model.get_episode(env)
        generated_y = model.generate(episode_x)
        psnr_j = 0
        ssim_j = 0
        for i in range(21):
            im1 = np.transpose(y[i].cpu().numpy().clip(0, 1) * 255, (1, 2, 0)).astype(np.uint8)
            im2 = np.transpose(generated_y[i].cpu().numpy().clip(0, 1) * 255, (1, 2, 0)).astype(np.uint8)
            psnr_j += cv2.PSNR(im1, im2)
            ssim_j += compare_ssim(im1, im2, multichannel=True)
        psnr_j /= 21
        ssim_j /= 21
        psnr.append(psnr_j)
        ssim.append(ssim_j)
    return np.mean(psnr), np.std(psnr), np.mean(ssim), np.std(ssim)


multiview = False
trained_model = "multiview" if multiview else "baseline"
if multiview:
    model = Model(max_scale=4,
                  steps_per_scale=int(25e3),
                  lr=1e-3,
                  multiview=True)
    model.load(f"../trained_models/{trained_model}")
    data = MultiViewData(episode_duration=21,
                         data_buffer_size=int(16e3),
                         batch_size=16)
    data.load(data_folder="../data/")
else:
    model = Model(max_scale=4,
                  steps_per_scale=int(25e3),
                  lr=1e-3)
    model.load(f"../trained_models/{trained_model}")
    data = ObservationData(obs_buffer_size=21,
                           data_buffer_size=int(16e3),
                           batch_size=16)
    data.load(data_folder="../data/")
env = Env(obs_buffer_size=21)

psnr_mean, psnr_std, ssim_mean, ssim_std = get_results(train=True)
print(f"{trained_model} training, PSNR:{psnr_mean} ({psnr_std}), SSIM:{ssim_mean} ({ssim_std})")

psnr_mean, psnr_std, ssim_mean, ssim_std = get_results(train=False)
print(f"{trained_model} testing, PSNR:{psnr_mean} ({psnr_std}), SSIM:{ssim_mean} ({ssim_std})")
