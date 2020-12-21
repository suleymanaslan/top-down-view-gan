import io
import numpy as np
from PIL import Image
from apng import APNG
import imageio


def save_images(train=True):
    im = APNG.open(f"../results/baseline/{'anim_train' if train else 'anim_test'}.png")

    for offset in range(2):
        im1 = np.concatenate(
            (np.array(Image.open(io.BytesIO(im.frames[0 + offset * 21][0].to_bytes())))[:68, :68, :],
             np.array(Image.open(io.BytesIO(im.frames[6 + offset * 21][0].to_bytes())))[:68, :68, :],
             np.array(Image.open(io.BytesIO(im.frames[12 + offset * 21][0].to_bytes())))[:68, :68, :],
             np.array(Image.open(io.BytesIO(im.frames[18 + offset * 21][0].to_bytes())))[:68, :68, :],
             ), axis=1)

        im2 = np.concatenate(
            (np.array(Image.open(io.BytesIO(im.frames[0 + offset * 21][0].to_bytes())))[:68, 68:136, :],
             np.array(Image.open(io.BytesIO(im.frames[6 + offset * 21][0].to_bytes())))[:68, 68:136, :],
             np.array(Image.open(io.BytesIO(im.frames[12 + offset * 21][0].to_bytes())))[:68, 68:136, :],
             np.array(Image.open(io.BytesIO(im.frames[18 + offset * 21][0].to_bytes())))[:68, 68:136, :],
             ), axis=1)

        imageio.imsave(f"../results/baseline/{'train' if train else 'test'}_x_{offset}.png", im1)
        imageio.imsave(f"../results/baseline/{'train' if train else 'test'}_g_{offset}.png", im2)


save_images(train=True)
save_images(train=False)
