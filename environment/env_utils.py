import numpy as np
import matplotlib.pyplot as plt
from skimage import transform, util
from IPython import display
import cv2


def get_view(env):
    view = env.render("rgb_array")
    view = util.img_as_ubyte(transform.resize(view, (128, 128), order=0))
    return view


def plt_show(im, figsize=(6, 6)):
    plt.figure(figsize=figsize)
    plt.clf()
    plt.imshow(im)
    plt.axis('off')
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.close()


def show_view(env):
    view = get_view(env)
    plt_show(view)


def get_zx(env):
    x = int((env.agent.pos * 133.0 / 12 - 2.5)[0])
    z = int((env.agent.pos * 133.0 / 12 - 2.5)[2])
    return z, x


def get_top_view(env):
    top_view = env.render("rgb_array", view='top')[52:547, 152:647, :]
    top_view = util.img_as_ubyte(transform.resize(top_view, (128, 128), order=0))
    z, x = get_zx(env)
    top_view[z, x] = 0
    return top_view


def show_top_view(env):
    top_view = get_top_view(env)
    plt_show(top_view)


def get_view_mask(env):
    top_view = get_top_view(env)
    view_mask = np.zeros_like(top_view)
    z, x = get_zx(env)
    view_mask = cv2.ellipse(view_mask, (x, z), axes=(180, 180), angle=env.agent.dir * -57.1428 - 45,
                            startAngle=-7, endAngle=98, color=(255, 255, 255), thickness=-1)
    return view_mask.astype(np.uint8)


def show_view_mask(env):
    view_mask = get_view_mask(env)
    plt_show(view_mask)


def get_masked_view(env):
    view_mask = get_view_mask(env)
    top_view = get_top_view(env)
    masked_view = np.where(view_mask != 0, top_view, 0)
    return masked_view


def show_masked_view(env):
    masked_view = get_masked_view(env)
    plt_show(masked_view)


def show_all_view(env):
    view = get_view(env)
    view_mask = get_view_mask(env)
    top_view = get_top_view(env)
    masked_view = get_masked_view(env)
    all_view = np.concatenate((view, view_mask, top_view, masked_view), axis=1)
    plt_show(all_view, figsize=(16, 16))


def show_observable_view(env):
    view = get_view(env)
    masked_view = get_masked_view(env)
    all_view = np.concatenate((view, masked_view), axis=1)
    plt_show(all_view, figsize=(10, 10))


def get_data(env):
    return get_view(env), get_masked_view(env)
