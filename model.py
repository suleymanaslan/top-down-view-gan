# adapted from https://github.com/facebookresearch/pytorch_GAN_zoo

import os
import time
import imageio
import torch
import cv2 as cv
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from datetime import datetime
from network import Generator, Discriminator
from network_utils import WGANGP, wgangp_gradient_penalty, finite_check


class Model:
    def __init__(self, max_scale=4, steps_per_scale=int(25e3), batch_size=64, lr=1e-3):
        self.device = torch.device("cuda:0")
        self.max_scale = max_scale
        self.image_size = 2 ** (self.max_scale + 2)
        self.steps_per_scale = steps_per_scale
        self.batch_size = batch_size
        self.lr = lr

        self.model_dir = None
        self.generated_random = None
        self.generated_fixed = None

        self.scale = 0
        self.steps = 0

        self.alpha = 0
        self.update_alpha_step = 25
        self.alpha_update_cons = 2 * self.update_alpha_step / self.steps_per_scale

        self.epsilon_d = 0.001

        self._init_networks()
        self._init_optimizers()
        self.loss_criterion = WGANGP(self.device)

        self.fixed_noise = torch.randn(self.batch_size, self.generator.dim_latent, device=self.device)

    def _init_networks(self):
        self.generator = Generator().to(self.device)
        self.discriminator = Discriminator().to(self.device)

    def _init_optimizers(self):
        self.optimizer_g = optim.Adam(filter(lambda p: p.requires_grad, self.generator.parameters()),
                                      betas=(0, 0.99), lr=self.lr)

        self.optimizer_d = optim.Adam(filter(lambda p: p.requires_grad, self.discriminator.parameters()),
                                      betas=(0, 0.99), lr=self.lr)

    def print_and_log(self, text):
        print(text)
        print(text, file=open(f'{self.model_dir}/log.txt', 'a'))

    def _init_training(self):
        training_timestamp = str(int(time.time()))
        self.model_dir = f'trained_models/model_{training_timestamp}/'

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        self.print_and_log(f"{datetime.now()}, start training")

    def save(self):
        if os.path.exists(self.model_dir):
            torch.save(self.generator.state_dict(), f"{self.model_dir}/generator.pth")
            torch.save(self.discriminator.state_dict(), f"{self.model_dir}/discriminator.pth")

    def load(self, load_dir):
        self.generator.load_state_dict(torch.load(f"{load_dir}/generator.pth"))
        self.discriminator.load_state_dict(torch.load(f"{load_dir}/discriminator.pth"))

    def _resize_concat(self, img_array):
        return np.concatenate((cv.resize(img_array[0], (self.image_size, self.image_size),
                                         interpolation=cv.INTER_NEAREST),
                               cv.resize(img_array[1], (self.image_size, self.image_size),
                                         interpolation=cv.INTER_NEAREST)),
                              axis=1)

    def _get_img_grid(self, img_array):
        top_img = self._resize_concat(img_array[:2])
        bot_img = self._resize_concat(img_array[2:4])
        return np.concatenate((top_img, bot_img), axis=0)

    def save_generated(self):
        if self.generated_random is None or self.generated_fixed is None:
            return
        generated_random = (self.generated_random[:4].permute(0, 2, 3, 1).detach().cpu().numpy().clip(0, 1)
                            * 255).astype(np.uint8)
        generated_fixed = (self.generated_fixed[:4].permute(0, 2, 3, 1).detach().cpu().numpy().clip(0, 1)
                           * 255).astype(np.uint8)
        img_grid_random = self._get_img_grid(generated_random)
        img_grid_fixed = self._get_img_grid(generated_fixed)
        write_time = int(time.time())
        imageio.imwrite(f"{self.model_dir}/{write_time}_fixed.jpg", img_grid_fixed)
        imageio.imwrite(f"{self.model_dir}/{write_time}_random.jpg", img_grid_random)

    def train(self, dataloader):
        if self.model_dir is None:
            self._init_training()

        steps = self.steps_per_scale * 8
        for step_i in range(steps):
            data = next(iter(dataloader))
            real_batch = data[0].to(self.device)
            self.train_step(real_batch)

    def train_step(self, real_batch):
        self.steps += 1

        if self.steps % self.update_alpha_step == 0 and self.alpha > 0:
            self.alpha = max(0.0, self.alpha - self.alpha_update_cons)

        if self.scale < self.max_scale:
            real_batch = F.avg_pool2d(real_batch, (2, 2))
            for _ in range(1, self.max_scale - self.scale):
                real_batch = F.avg_pool2d(real_batch, (2, 2))

        if self.alpha > 0:
            low_res_real = F.avg_pool2d(real_batch, (2, 2))
            low_res_real = F.interpolate(low_res_real, scale_factor=2, mode='nearest')
            real_batch = self.alpha * low_res_real + (1 - self.alpha) * real_batch

        self.generator.set_alpha(self.alpha)
        self.discriminator.set_alpha(self.alpha)

        self.optimizer_d.zero_grad()

        pred_real_d = self.discriminator(real_batch)
        loss_d = self.loss_criterion.get_criterion(pred_real_d, True)
        all_loss_d = loss_d

        input_latent = torch.randn(self.batch_size, self.generator.dim_latent, device=self.device)

        pred_fake_g = self.generator(input_latent).detach()
        pred_fake_d = self.discriminator(pred_fake_g, False)
        loss_d_fake = self.loss_criterion.get_criterion(pred_fake_d, False)
        all_loss_d += loss_d_fake

        loss_d_grad = wgangp_gradient_penalty(real_batch, pred_fake_g, self.discriminator, weight=10.0, backward=True)

        loss_epsilon = (pred_real_d[:, 0] ** 2).sum() * self.epsilon_d
        all_loss_d += loss_epsilon

        all_loss_d.backward(retain_graph=True)
        finite_check(self.discriminator.parameters())
        self.optimizer_d.step()

        self.optimizer_g.zero_grad()

        input_noise = torch.randn(self.batch_size, self.generator.dim_latent, device=self.device)

        pred_fake_g = self.generator(input_noise)

        pred_fake_d, phi_g_fake = self.discriminator(pred_fake_g, True)
        loss_g_fake = self.loss_criterion.get_criterion(pred_fake_d, True)
        loss_g_fake.backward(retain_graph=True)
        finite_check(self.generator.parameters())
        self.optimizer_g.step()

        loss_dict = {"g_fake": loss_g_fake,
                     "d_real": loss_d,
                     "d_fake": loss_d_fake,
                     "d_grad": loss_d_grad,
                     "epsilon": loss_epsilon}

        if self.steps == 1 or self.steps % (self.steps_per_scale // 10) == 0:
            self.print_and_log(f"{datetime.now()} "
                               f"[{self.scale}/{self.max_scale}][{self.steps:06d}], "
                               f"A:{self.alpha:.2f}, L_G:{loss_dict['g_fake'].item():.2f}, "
                               f"L_DR:{loss_dict['d_real'].item():.2f}, L_DF:{loss_dict['d_fake'].item():.2f}, "
                               f"L_DG:{loss_dict['d_grad']:.2f}, L_DE:{loss_dict['epsilon'].item():.2f}")
            self.generated_random = pred_fake_g
            self.generated_fixed = self.generator(self.fixed_noise)
            self.save_generated()

        if self.steps % self.steps_per_scale == 0:
            if self.scale < self.max_scale:
                self.generator.add_scale(depth_new_scale=512)
                self.discriminator.add_scale(depth_new_scale=512)

                self.generator.to(self.device)
                self.discriminator.to(self.device)

                self._init_optimizers()

                self.scale += 1
                self.alpha = 1.0
