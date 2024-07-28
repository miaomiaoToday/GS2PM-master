"""
Implementations of Masked Spatiotemporal Sequence Pretraining with Learnable Prompting Embedding.
The model should own 2 modes: one for pretraining and another for tuning. Specially for tunning, only the prompting
embedding should be tuned.
"""

import torch
from torch import nn

from .MS2P_modules import (ConvSC, gInception_ST, UniformerSubBlock)
import numpy as np


class PadPrompter(nn.Module):
    def __init__(self, prompt_size, crop_size):
        super(PadPrompter, self).__init__()
        pad_size = prompt_size
        image_size = crop_size

        self.base_size = image_size - pad_size * 2
        self.pad_up = nn.Parameter(torch.randn([1, 3, pad_size, image_size]))
        self.pad_down = nn.Parameter(torch.randn([1, 3, pad_size, image_size]))
        self.pad_left = nn.Parameter(torch.randn([1, 3, image_size-pad_size*2, pad_size]))
        self.pad_right = nn.Parameter(torch.randn([1, 3, image_size-pad_size*2, pad_size]))

    def forward(self, x):
        base = torch.zeros(1, 3, self.base_size, self.base_size).to(x.device)
        # base = torch.zeros(1, 3, self.base_size, self.base_size)
        prompt = torch.cat([self.pad_left, base, self.pad_right], dim=3)
        prompt = torch.cat([self.pad_up, prompt, self.pad_down], dim=2)
        # temp = x.size(0)
        # temp1 = [prompt]
        # temp2 = temp * temp1
        prompt = torch.cat(x.size(0)*[prompt])

        return x + prompt


class MaskGenerator(nn.Module):
    def __init__(self, input_size=192, mask_patch_size=32, model_patch_size=4, mask_ratio=0.6):
        super(MaskGenerator, self).__init__()
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio

        assert self.input_size % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0

        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size

        self.token_count = self.rand_size ** 2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))

    def forward(self, x):
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1

        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)

        # to tensor and cuda
        mask = torch.tensor(mask)
        mask = mask.to(x.device)
        # to repeat into [128, 128] -> [10, 3, 128, 128]
        mask = mask.repeat(3, 1, 1)
        mask = mask.repeat(x.size(0), 1, 1, 1)

        y = x * mask

        return y


class MS2Pv3_tune(nn.Module):
    r"""SimVP Model

    Implementation of `SimVP: Simpler yet Better Video Prediction
    <https://arxiv.org/abs/2206.05099>`_.

    """

    def __init__(self, in_shape, hid_S=16, hid_T=256, N_S=4, N_T=4, model_type='gSTA',
                 mlp_ratio=8., drop=0.0, drop_path=0.0, spatio_kernel_enc=3,
                 spatio_kernel_dec=3, act_inplace=True, **kwargs):
        super(MS2Pv3_tune, self).__init__()
        T, C, H, W = in_shape  # T is pre_seq_length
        H, W = int(H / 2 ** (N_S / 2)), int(W / 2 ** (N_S / 2))  # downsample 1 / 2**(N_S/2)
        act_inplace = False

        # siamese encoder (waiting for modification E3)
        self.enc = Encoder(C, hid_S, N_S, spatio_kernel_enc, act_inplace=act_inplace)
        # siamese decoder
        self.dec = Decoder(hid_S, C, N_S, spatio_kernel_dec, act_inplace=act_inplace)
        # mid translator
        self.translator = MidMetaNet(T * hid_S, hid_T, N_T,
                                     input_resolution=(H, W), model_type=model_type,
                                     mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)
        # mask generator
        self.mask = MaskGenerator(input_size=128, mask_patch_size=16, model_patch_size=1, mask_ratio=0.6)
        # visual prompt generator
        self.prompt = PadPrompter(32, 128)


    def forward(self, x, **kwargs):
        """1. upper branch, task, video prediction. [input + prompt] -> enc -> mid -> dec -> output"""
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        # x + prompt -> x
        x_upper = self.prompt(x)
        # siamese encoding
        embed_upper, skip_upper = self.enc(x_upper)
        _, C_, H_, W_ = embed_upper.shape
        z = embed_upper.view(B, T, C_, H_, W_)
        z = self.translator(z)
        z = z.reshape(B * T, C_, H_, W_)
        # siamese decoding
        Y_pred = self.dec(z, skip_upper)
        Y_pred = Y_pred.reshape(B, T, C, H, W)

        """2. lower branch: auto encoding. [input * mask] -> enc -> dec -> input"""
        # ignore

        return Y_pred


def sampling_generator(N, reverse=False):
    samplings = [False, True] * (N // 2)
    if reverse:
        return list(reversed(samplings[:N]))
    else:
        return samplings[:N]


class Encoder(nn.Module):
    """3D Encoder for SimVP"""

    def __init__(self, C_in, C_hid, N_S, spatio_kernel, act_inplace=True):
        samplings = sampling_generator(N_S)
        super(Encoder, self).__init__()
        self.enc = nn.Sequential(
            ConvSC(C_in, C_hid, spatio_kernel, downsampling=samplings[0],
                   act_inplace=act_inplace),
            *[ConvSC(C_hid, C_hid, spatio_kernel, downsampling=s,
                     act_inplace=act_inplace) for s in samplings[1:]]
        )

    def forward(self, x):  # B*4, 3, 128, 128
        enc1 = self.enc[0](x)
        latent = enc1
        for i in range(1, len(self.enc)):
            latent = self.enc[i](latent)
        return latent, enc1


class Decoder(nn.Module):
    """3D Decoder for SimVP"""

    def __init__(self, C_hid, C_out, N_S, spatio_kernel, act_inplace=True):
        samplings = sampling_generator(N_S, reverse=True)
        super(Decoder, self).__init__()
        self.dec = nn.Sequential(
            *[ConvSC(C_hid, C_hid, spatio_kernel, upsampling=s,
                     act_inplace=act_inplace) for s in samplings[:-1]],
            ConvSC(C_hid, C_hid, spatio_kernel, upsampling=samplings[-1],
                   act_inplace=act_inplace)
        )
        self.readout = nn.Conv2d(C_hid, C_out, 1)

    def forward(self, hid, enc1=None):
        for i in range(0, len(self.dec) - 1):
            hid = self.dec[i](hid)
        Y = self.dec[-1](hid + enc1)
        Y = self.readout(Y)
        return Y


class MidIncepNet(nn.Module):
    """The hidden Translator of IncepNet for SimVPv1"""

    def __init__(self, channel_in, channel_hid, N2, incep_ker=[3, 5, 7, 11], groups=8, **kwargs):
        super(MidIncepNet, self).__init__()
        assert N2 >= 2 and len(incep_ker) > 1
        self.N2 = N2
        enc_layers = [gInception_ST(
            channel_in, channel_hid // 2, channel_hid, incep_ker=incep_ker, groups=groups)]
        for i in range(1, N2 - 1):
            enc_layers.append(
                gInception_ST(channel_hid, channel_hid // 2, channel_hid,
                              incep_ker=incep_ker, groups=groups))
        enc_layers.append(
            gInception_ST(channel_hid, channel_hid // 2, channel_hid,
                          incep_ker=incep_ker, groups=groups))
        dec_layers = [
            gInception_ST(channel_hid, channel_hid // 2, channel_hid,
                          incep_ker=incep_ker, groups=groups)]
        for i in range(1, N2 - 1):
            dec_layers.append(
                gInception_ST(2 * channel_hid, channel_hid // 2, channel_hid,
                              incep_ker=incep_ker, groups=groups))
        dec_layers.append(
            gInception_ST(2 * channel_hid, channel_hid // 2, channel_in,
                          incep_ker=incep_ker, groups=groups))

        self.enc = nn.Sequential(*enc_layers)
        self.dec = nn.Sequential(*dec_layers)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B, T * C, H, W)

        # encoder
        skips = []
        z = x
        for i in range(self.N2):
            z = self.enc[i](z)
            if i < self.N2 - 1:
                skips.append(z)
        # decoder
        z = self.dec[0](z)
        for i in range(1, self.N2):
            z = self.dec[i](torch.cat([z, skips[-i]], dim=1))

        y = z.reshape(B, T, C, H, W)
        return y


class MetaBlock(nn.Module):
    """The hidden Translator of MetaFormer for SimVP"""

    def __init__(self, in_channels, out_channels, input_resolution=None, model_type=None,
                 mlp_ratio=8., drop=0.0, drop_path=0.0, layer_i=0):
        super(MetaBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # only do uniformer as our translation module.
        block_type = 'MHSA' if in_channels == out_channels and layer_i > 0 else 'Conv'
        self.block = UniformerSubBlock(
            in_channels, mlp_ratio=mlp_ratio, drop=drop,
            drop_path=drop_path, block_type=block_type)

        if in_channels != out_channels:
            self.reduction = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        z = self.block(x)
        return z if self.in_channels == self.out_channels else self.reduction(z)


"""for prompt"""


class MetaBlock_target(nn.Module):
    """The hidden Translator of MetaFormer for SimVP"""

    def __init__(self, in_channels, out_channels, input_resolution=None, model_type=None,
                 mlp_ratio=8., drop=0.0, drop_path=0.0, layer_i=0):
        super(MetaBlock_target, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # only do uniformer as our translation module.
        self.block_type = 'MHSAQKV' if in_channels == out_channels and layer_i > 0 else 'Conv'
        self.block = UniformerSubBlock(
            in_channels, mlp_ratio=mlp_ratio, drop=drop,
            drop_path=drop_path, block_type=self.block_type)

        if in_channels != out_channels:
            self.reduction = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, q, k, v):
        if self.block_type == 'Conv':
            z = self.block(q)
        elif self.block_type == 'MHSAQKV':
            z = self.block(q, k, v)
        return z if self.in_channels == self.out_channels else self.reduction(z)


class MidMetaNet(nn.Module):
    """The hidden Translator of MetaFormer for SimVP"""

    def __init__(self, channel_in, channel_hid, N2,
                 input_resolution=None, model_type=None,
                 mlp_ratio=4., drop=0.0, drop_path=0.1):
        super(MidMetaNet, self).__init__()
        assert N2 >= 2 and mlp_ratio > 1
        self.N2 = N2
        dpr = [  # stochastic depth decay rule
            x.item() for x in torch.linspace(1e-2, drop_path, self.N2)]

        # downsample
        enc_layers = [MetaBlock(
            channel_in, channel_hid, input_resolution, model_type,
            mlp_ratio, drop, drop_path=dpr[0], layer_i=0)]
        # middle layers
        for i in range(1, N2 - 1):
            enc_layers.append(MetaBlock(
                channel_hid, channel_hid, input_resolution, model_type,
                mlp_ratio, drop, drop_path=dpr[i], layer_i=i))
        # upsample
        enc_layers.append(MetaBlock(
            channel_hid, channel_in, input_resolution, model_type,
            mlp_ratio, drop, drop_path=drop_path, layer_i=N2 - 1))
        self.enc = nn.Sequential(*enc_layers)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B, T * C, H, W)

        z = x
        for i in range(self.N2):
            z = self.enc[i](z)

        y = z.reshape(B, T, C, H, W)
        return y


"""for prompt"""


class MidMetaNet_prompt(nn.Module):
    """The hidden Translator of MetaFormer for SimVP"""

    def __init__(self, channel_in, channel_hid, N2,
                 input_resolution=None, model_type=None,
                 mlp_ratio=4., drop=0.0, drop_path=0.1):
        super(MidMetaNet_prompt, self).__init__()
        assert N2 >= 2 and mlp_ratio > 1
        self.N2 = N2
        dpr = [  # stochastic depth decay rule
            x.item() for x in torch.linspace(1e-2, drop_path, self.N2)]

        # downsample
        enc_layers = [MetaBlock(
            channel_in, channel_hid, input_resolution, model_type,
            mlp_ratio, drop, drop_path=dpr[0], layer_i=0)]
        # middle layers
        for i in range(1, N2 - 1):
            enc_layers.append(MetaBlock(
                channel_hid, channel_hid, input_resolution, model_type,
                mlp_ratio, drop, drop_path=dpr[i], layer_i=i))
        """ignore upsample, just do as an encoder"""
        # # upsample, out channel hid for make with the target branch
        # enc_layers.append(MetaBlock(
        #     channel_hid, channel_hid, input_resolution, model_type,
        #     mlp_ratio, drop, drop_path=drop_path, layer_i=N2-1))
        self.enc = nn.Sequential(*enc_layers)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B, T * C, H, W)

        z = x
        # without last layer
        for i in range(self.N2 - 1):
            z = self.enc[i](z)

        # y = z.reshape(B, T, C, H, W)
        # B, 128(mid_dim), H, W for make up in decoder
        return z


"""for target intrageting prompt"""


class MidMetaNet_target(nn.Module):
    """The hidden Translator of MetaFormer for SimVP"""

    def __init__(self, channel_in, channel_hid, N2,
                 input_resolution=None, model_type=None,
                 mlp_ratio=4., drop=0.0, drop_path=0.1):
        super(MidMetaNet_target, self).__init__()
        assert N2 >= 2 and mlp_ratio > 1
        self.N2 = N2
        dpr = [  # stochastic depth decay rule
            x.item() for x in torch.linspace(1e-2, drop_path, self.N2)]

        # downsample
        enc_layers = [MetaBlock_target(
            channel_in, channel_hid, input_resolution, model_type,
            mlp_ratio, drop, drop_path=dpr[0], layer_i=0)]
        # middle layers
        for i in range(1, N2 - 1):
            enc_layers.append(MetaBlock_target(
                channel_hid, channel_hid, input_resolution, model_type,
                mlp_ratio, drop, drop_path=dpr[i], layer_i=i))
        # upsample
        enc_layers.append(MetaBlock_target(
            channel_hid, channel_in, input_resolution, model_type,
            mlp_ratio, drop, drop_path=drop_path, layer_i=N2 - 1))
        self.enc = nn.Sequential(*enc_layers)

    def forward(self, q, k, v):
        B, T, C, H, W = q.shape
        # (1) same treatment with q k v, and (2) well transmit q
        q = q.reshape(B, T * C, H, W)
        # k = k.reshape(B, T*C, H, W)   # k and v are already in encoder.
        # v = v.reshape(B, T*C, H, W)

        for i in range(self.N2):
            q = self.enc[i](q, k, v)

        y = q.reshape(B, T, C, H, W)
        return y
