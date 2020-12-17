# adapted from https://github.com/facebookresearch/pytorch_GAN_zoo

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.layers import flatten, upscale2d, EqualizedLinear, EqualizedConv2d, NormalizationLayer
from model.network_utils import mini_batch_std_dev


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = None if stride == 1 else nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride,
                                                             bias=False)
        self.downsample_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
            identity = self.downsample_bn(identity)
        out += identity
        out = self.relu(out)
        return out


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.block = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, 1, 1),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Conv2d(out_channels, out_channels, 3, 1, 1),
                                   nn.LeakyReLU(0.2, inplace=True), nn.AvgPool2d(2),
                                   )

    def forward(self, x):
        return self.block(x)


class Encoder3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=3, padding=1, pool_kernel=2):
        super(Encoder3DBlock, self).__init__()
        self.block = nn.Sequential(nn.Conv3d(in_channels, out_channels, 3, 1, 1),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Conv3d(out_channels, out_channels, kernel, 1, padding),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.AvgPool3d(pool_kernel),
                                   )

    def forward(self, x):
        return self.block(x)


class EncoderResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderResNetBlock, self).__init__()
        self.block = nn.Sequential(BasicBlock(in_channels, in_channels, stride=1),
                                   BasicBlock(in_channels, out_channels, stride=2),
                                   )

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.in_channel = 3 * 21
        self.net = nn.Sequential(nn.Conv2d(self.in_channel, 64, 1, 1, 0), nn.LeakyReLU(0.2, inplace=True),
                                 EncoderBlock(64, 128),
                                 EncoderBlock(128, 128),
                                 EncoderBlock(128, 256),
                                 EncoderBlock(256, 256),
                                 )
        self.out_dim = 256 * 4 * 4

    def forward(self, x):
        x = x.view(x.shape[0], self.in_channel, 64, 64)
        return self.net(x).view(x.shape[0], self.out_dim)


class SpatioTemporalEncoder(nn.Module):
    def __init__(self):
        super(SpatioTemporalEncoder, self).__init__()
        self.depth = 21
        self.spatial_net = nn.Sequential(nn.Conv2d(3, 32, 1, 1, 0), nn.LeakyReLU(0.2, inplace=True),
                                         EncoderResNetBlock(32, 64),
                                         EncoderResNetBlock(64, 64),
                                         EncoderResNetBlock(64, 128),
                                         EncoderResNetBlock(128, 128),
                                         )
        self.temporal_net = nn.Sequential(nn.Conv1d(2048, 2048, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True),
                                          nn.Conv1d(2048, 2048, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True),
                                          nn.Conv1d(2048, 4096, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True),
                                          nn.Conv1d(4096, 4096, 3, 1, 0), nn.LeakyReLU(0.2, inplace=True),
                                          )
        self.spatial_out_dim = 128 * 4 * 4
        self.out_dim = 4096

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(batch_size * self.depth, 3, 64, 64)
        x = self.spatial_net(x).view(batch_size, self.depth, self.spatial_out_dim)
        x = x.permute(0, 2, 1).contiguous()
        x = self.temporal_net(x).view(batch_size, self.out_dim)
        return x


class Encoder3D(nn.Module):
    def __init__(self):
        super(Encoder3D, self).__init__()
        self.depth = 21
        self.net = nn.Sequential(Encoder3DBlock(3, 32, kernel=[2, 3, 3], padding=[0, 1, 1]),
                                 Encoder3DBlock(32, 64),
                                 Encoder3DBlock(64, 128, kernel=[2, 3, 3], padding=[0, 1, 1]),
                                 Encoder3DBlock(128, 256, kernel=[2, 3, 3], padding=[0, 1, 1], pool_kernel=[1, 2, 2]),
                                 )
        self.out_dim = 256 * 4 * 4

    def forward(self, x):
        return self.net(x).view(x.shape[0], self.out_dim)


class MultiViewEncoder(nn.Module):
    def __init__(self):
        super(MultiViewEncoder, self).__init__()
        self.net_first_person = nn.Sequential(nn.Conv2d(3, 32, 1, 1, 0), nn.LeakyReLU(0.2, inplace=True),
                                              EncoderBlock(32, 64),
                                              EncoderBlock(64, 64),
                                              EncoderBlock(64, 128),
                                              EncoderBlock(128, 128),
                                              )
        self.net_top_down = nn.Sequential(nn.Conv2d(3, 32, 1, 1, 0), nn.LeakyReLU(0.2, inplace=True),
                                          EncoderBlock(32, 64),
                                          EncoderBlock(64, 64),
                                          EncoderBlock(64, 128),
                                          EncoderBlock(128, 128),
                                          )
        self.out_dim = 256 * 4 * 4

    def forward(self, x):
        x_fpv = x[:, :, 0, :, :]
        x_tdv = x[:, :, 1, :, :]
        feat_fpv = self.net_first_person(x_fpv).view(x_fpv.shape[0], self.out_dim // 2)
        feat_tdv = self.net_first_person(x_tdv).view(x_tdv.shape[0], self.out_dim // 2)
        return torch.cat((feat_fpv, feat_tdv), dim=1)


class Generator(nn.Module):
    def __init__(self, encoder):
        super(Generator, self).__init__()
        self.encoder = encoder
        self.dim_latent = self.encoder.out_dim
        self.depth_scale0 = 256
        self.dim_output = 3
        self.equalized_lr = True
        self.init_bias_to_zero = True
        self.scales_depth = [self.depth_scale0]

        self.scale_layers = nn.ModuleList()

        self.to_rgb_layers = nn.ModuleList()
        self.to_rgb_layers.append(EqualizedConv2d(self.depth_scale0, self.dim_output, 1, equalized=self.equalized_lr,
                                                  init_bias_to_zero=self.init_bias_to_zero))

        self.format_layer = EqualizedLinear(self.dim_latent, 16 * self.scales_depth[0], equalized=self.equalized_lr,
                                            init_bias_to_zero=self.init_bias_to_zero)

        self.group_scale0 = nn.ModuleList()
        self.group_scale0.append(
            EqualizedConv2d(self.depth_scale0, self.depth_scale0, 3, padding=1, equalized=self.equalized_lr,
                            init_bias_to_zero=self.init_bias_to_zero))

        self.alpha = 0

        self.leaky_relu = torch.nn.LeakyReLU(0.2, inplace=True)

        self.normalization_layer = NormalizationLayer()

        self.generation_activation = None

    def add_scale(self, depth_new_scale=None):
        if depth_new_scale is None:
            depth_new_scale = self.depth_scale0
        depth_last_scale = self.scales_depth[-1]
        self.scales_depth.append(depth_new_scale)

        self.scale_layers.append(nn.ModuleList())
        self.scale_layers[-1].append(
            EqualizedConv2d(depth_last_scale, depth_new_scale, 3, padding=1, equalized=self.equalized_lr,
                            init_bias_to_zero=self.init_bias_to_zero))
        self.scale_layers[-1].append(
            EqualizedConv2d(depth_new_scale, depth_new_scale, 3, padding=1, equalized=self.equalized_lr,
                            init_bias_to_zero=self.init_bias_to_zero))

        self.to_rgb_layers.append(EqualizedConv2d(depth_new_scale, self.dim_output, 1, equalized=self.equalized_lr,
                                                  init_bias_to_zero=self.init_bias_to_zero))

    def set_alpha(self, alpha):
        self.alpha = alpha

    def forward(self, x):
        x = self.encoder(x)

        x = self.normalization_layer(x)
        x = flatten(x)
        x = self.leaky_relu(self.format_layer(x))
        x = x.view(x.size()[0], -1, 4, 4)
        x = self.normalization_layer(x)

        for conv_layer in self.group_scale0:
            x = self.leaky_relu(conv_layer(x))
            x = self.normalization_layer(x)

        if self.alpha > 0 and len(self.scale_layers) == 1:
            y = self.to_rgb_layers[-2](x)
            y = upscale2d(y)

        for scale, layer_group in enumerate(self.scale_layers, 0):
            x = upscale2d(x)
            for conv_layer in layer_group:
                x = self.leaky_relu(conv_layer(x))
                x = self.normalization_layer(x)
            if self.alpha > 0 and scale == (len(self.scale_layers) - 2):
                y = self.to_rgb_layers[-2](x)
                y = upscale2d(y)

        x = self.to_rgb_layers[-1](x)

        if self.alpha > 0:
            x = self.alpha * y + (1.0 - self.alpha) * x

        if self.generation_activation is not None:
            x = self.generation_activation(x)

        return x


class Discriminator(nn.Module):
    def __init__(self, feat_dim):
        super(Discriminator, self).__init__()
        self.dim_input = 3
        self.depth_scale0 = 256
        self.size_decision_layer = 1
        self.equalized_lr = True
        self.init_bias_to_zero = True
        self.mini_batch_normalization = True
        self.dim_entry_scale0 = self.depth_scale0 + 1
        self.scales_depth = [self.depth_scale0]

        self.scale_layers = nn.ModuleList()

        self.from_rgb_layers = nn.ModuleList()
        self.from_rgb_layers.append(EqualizedConv2d(self.dim_input, self.depth_scale0, 1, equalized=self.equalized_lr,
                                                    init_bias_to_zero=self.init_bias_to_zero))

        self.merge_layers = nn.ModuleList()

        self.decision_layer = EqualizedLinear(self.scales_depth[0], self.size_decision_layer,
                                              equalized=self.equalized_lr, init_bias_to_zero=self.init_bias_to_zero)

        self.group_scale0 = nn.ModuleList()
        self.group_scale0.append(
            EqualizedConv2d(self.dim_entry_scale0, self.depth_scale0, 3, padding=1, equalized=self.equalized_lr,
                            init_bias_to_zero=self.init_bias_to_zero))
        self.group_scale0.append(
            EqualizedLinear(self.depth_scale0 * 16 + feat_dim, self.depth_scale0, equalized=self.equalized_lr,
                            init_bias_to_zero=self.init_bias_to_zero))

        self.alpha = 0

        self.leaky_relu = torch.nn.LeakyReLU(0.2, inplace=True)

    def add_scale(self, depth_new_scale=None):
        if depth_new_scale is None:
            depth_new_scale = self.depth_scale0
        depth_last_scale = self.scales_depth[-1]
        self.scales_depth.append(depth_new_scale)

        self.scale_layers.append(nn.ModuleList())
        self.scale_layers[-1].append(
            EqualizedConv2d(depth_new_scale, depth_new_scale, 3, padding=1, equalized=self.equalized_lr,
                            init_bias_to_zero=self.init_bias_to_zero))
        self.scale_layers[-1].append(
            EqualizedConv2d(depth_new_scale, depth_last_scale, 3, padding=1, equalized=self.equalized_lr,
                            init_bias_to_zero=self.init_bias_to_zero))

        self.from_rgb_layers.append(
            EqualizedConv2d(self.dim_input, depth_new_scale, 1, equalized=self.equalized_lr,
                            init_bias_to_zero=self.init_bias_to_zero))

    def set_alpha(self, alpha):
        self.alpha = alpha

    def forward(self, fp_feat, td_view, get_feature=False):
        if self.alpha > 0 and len(self.from_rgb_layers) > 1:
            z = F.avg_pool2d(td_view, (2, 2))
            z = self.leaky_relu(self.from_rgb_layers[-2](z))

        td_view = self.leaky_relu(self.from_rgb_layers[-1](td_view))

        merge_layer = self.alpha > 0 and len(self.scale_layers) > 1

        shift = len(self.from_rgb_layers) - 2

        for group_layer in reversed(self.scale_layers):
            for layer in group_layer:
                td_view = self.leaky_relu(layer(td_view))

            td_view = nn.AvgPool2d((2, 2))(td_view)

            if merge_layer:
                merge_layer = False
                td_view = self.alpha * z + (1 - self.alpha) * td_view

            shift -= 1

        if self.mini_batch_normalization:
            td_view = mini_batch_std_dev(td_view)

        td_view = self.leaky_relu(self.group_scale0[0](td_view))

        td_view = flatten(td_view)
        td_view = torch.cat((td_view, fp_feat), dim=1)

        td_view = self.leaky_relu(self.group_scale0[1](td_view))

        out = self.decision_layer(td_view)

        if not get_feature:
            return out

        return out, td_view
