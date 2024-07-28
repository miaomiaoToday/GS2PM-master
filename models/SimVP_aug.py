import torch
from torch import nn
from .SimVP_modules import ConvSC, Inception
# from utils.grid import *
import torch.nn.functional as F

def stride_generator(N, reverse=False):
    strides = [1, 2] * 10
    if reverse:
        return list(reversed(strides[:N]))
    else:
        return strides[:N]


class Encoder(nn.Module):
    def __init__(self, C_in, C_hid, N_S):
        super(Encoder, self).__init__()
        strides = stride_generator(N_S)
        self.enc = nn.Sequential(
            ConvSC(C_in, C_hid, stride=strides[0]),
            *[ConvSC(C_hid, C_hid, stride=s) for s in strides[1:]]
        )

    def forward(self, x):  # B*4, 3, 128, 128
        enc1 = self.enc[0](x)
        latent = enc1
        for i in range(1, len(self.enc)):
            latent = self.enc[i](latent)
        return latent, enc1


class Decoder(nn.Module):
    def __init__(self, C_hid, C_out, N_S):
        super(Decoder, self).__init__()
        strides = stride_generator(N_S, reverse=True)
        self.dec = nn.Sequential(
            *[ConvSC(C_hid, C_hid, stride=s, transpose=True) for s in strides[:-1]],
            ConvSC(2 * C_hid, C_hid, stride=strides[-1], transpose=True)
        )
        self.readout = nn.Conv2d(C_hid, C_out, 1)

    def forward(self, hid, enc1=None):
        for i in range(0, len(self.dec) - 1):
            hid = self.dec[i](hid)
        Y = self.dec[-1](torch.cat([hid, enc1], dim=1))
        Y = self.readout(Y)
        return Y


class Mid_Xnet(nn.Module):
    def __init__(self, channel_in, channel_hid, N_T, incep_ker=[3, 5, 7, 11], groups=8):
        super(Mid_Xnet, self).__init__()

        self.N_T = N_T
        enc_layers = [Inception(channel_in, channel_hid // 2, channel_hid, incep_ker=incep_ker, groups=groups)]
        for i in range(1, N_T - 1):
            enc_layers.append(Inception(channel_hid, channel_hid // 2, channel_hid, incep_ker=incep_ker, groups=groups))
        enc_layers.append(Inception(channel_hid, channel_hid // 2, channel_hid, incep_ker=incep_ker, groups=groups))

        dec_layers = [Inception(channel_hid, channel_hid // 2, channel_hid, incep_ker=incep_ker, groups=groups)]
        for i in range(1, N_T - 1):
            dec_layers.append(
                Inception(2 * channel_hid, channel_hid // 2, channel_hid, incep_ker=incep_ker, groups=groups))
        dec_layers.append(Inception(2 * channel_hid, channel_hid // 2, channel_in, incep_ker=incep_ker, groups=groups))

        self.enc = nn.Sequential(*enc_layers)
        self.dec = nn.Sequential(*dec_layers)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B, T * C, H, W)

        # encoder
        skips = []
        z = x
        for i in range(self.N_T):
            z = self.enc[i](z)
            if i < self.N_T - 1:
                skips.append(z)

        # decoder
        z = self.dec[0](z)
        for i in range(1, self.N_T):
            z = self.dec[i](torch.cat([z, skips[-i]], dim=1))

        y = z.reshape(B, T, C, H, W)
        return y


class Branch(nn.Module):
    def __init__(self, feat_dim):
        super(Branch, self).__init__()
        self.fc1 = nn.Linear(65536, feat_dim)
        self.fc2 = nn.Linear(feat_dim, 65536)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, in_x, yaw, in_y):
        # flatten
        # in_x = torch.flatten(in_x, start_dim=1)
        Tf, Cf, Hf, Wf = in_x.shape
        in_x = in_x.view(Tf, Cf*Hf*Wf)
        x = self.fc1(in_x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)

        # yaw = yaw.view(yaw.size(0),1)
        # yaw = yaw.expand_as(x)

        in_y = in_y.view(Tf, Cf*Hf*Wf)
        feature = yaw * x + in_x      # for local improvemnet
        # feature = yaw * x + in_y        # for ood improv..

        feature = feature.view(Tf, Cf, Hf, Wf)

        return feature


class SimVP_aug(nn.Module):
    def __init__(self, shape_in, hid_S=16, hid_T=256, N_S=4, N_T=8, incep_ker=[3, 5, 7, 11], groups=8):
        super(SimVP_aug, self).__init__()
        T, C, H, W = shape_in
        self.enc = Encoder(C, hid_S, N_S)
        self.hid = Mid_Xnet(T * hid_S, hid_T, N_T, incep_ker, groups)
        self.dec = Decoder(hid_S, C, N_S)
        # aug
        self.feature_center = None
        # self.s_updater = Branch(feat_dim=64)
        self.s_updater = Branch(feat_dim=512)
        # self.linear1 = torch.nn.Linear(num_i, num_h)
        # self.relu = torch.nn.ReLU()
        # self.linear2 = torch.nn.Linear(num_h, num_h)  # 2个隐层
        # self.relu2 = torch.nn.ReLU()
        # self.linear3 = torch.nn.Linear(num_h, num_o)

    def space_update(self, f1, f2):
        if self.feature_center is None:
            self.feature_center = f1
        else:
            self.feature_center = (self.feature_center + f2) / 2.0

        feature1 = self.feature_center.view(self.feature_center.shape[0], -1)
        feature2 = f2.view(f2.shape[0], -1)

        feature1 = feature1.view(feature1.shape[0], -1)  # 将特征转换为N*(C*W*H)，即两维
        feature2 = feature2.view(feature2.shape[0], -1)
        feature1 = F.normalize(feature1)  # F.normalize只能处理两维的数据，L2归一化
        feature2 = F.normalize(feature2)
        distance = feature1.mm(feature2.t())  # 计算余弦相似度
        coef = torch.mean(distance)
        f_up = self.s_updater(f2, coef, f1)
        # torch.nn.Linear(num_i, num_h)
        return f_up



    def forward(self, x_raw, mode, grid=None):
        if mode == 'valid' or mode == 'test':
            B, T, C, H, W = x_raw.shape
            x = x_raw.view(B * T, C, H, W)

            embed, skip = self.enc(x)
            _, C_, H_, W_ = embed.shape

            z = embed.view(B, T, C_, H_, W_)
            hid = self.hid(z)
            # hid = self.hid( * z)
            hid = hid.reshape(B * T, C_, H_, W_)

            Y = self.dec(hid, skip)
            Y = Y.reshape(B, T, C, H, W)
            return Y
        elif mode == 'train':
            x_raw2 = torch.ones_like(x_raw)
            x_raw = x_raw

            for bs in range(x_raw.size(0)):
                # input2[bs, :, :, :, :] = grid(input2[bs, :, :, :, :])
                for ts in range(x_raw.size(1)):
                    x_raw2[bs, ts] = grid(x_raw[bs, ts])

            B, T, C, H, W = x_raw.shape
            x = x_raw.view(B * T, C, H, W)
            x2 = x_raw2.view(B * T, C, H, W)

            embed, skip = self.enc(x)
            embed2, skip2 = self.enc(x2)
            _, C_, H_, W_ = embed.shape

            embed2 = self.space_update(embed, embed2)

            # to get a coefficient to decide if the ood is okay, if not, ignore this pass
            # it should be A+ a*B not B+ a*B

            z = embed.view(B, T, C_, H_, W_)
            z2 = embed2.view(B, T, C_, H_, W_)
            hid = self.hid(z)
            hid2 = self.hid(z2)
            hid = hid.reshape(B * T, C_, H_, W_)
            hid2 = hid2.reshape(B * T, C_, H_, W_)

            hid2 = self.space_update(hid, hid2)

            Y = self.dec(hid, skip)
            Y2 = self.dec(hid2, skip2)
            Y = Y.reshape(B, T, C, H, W)
            Y2 = Y2.reshape(B, T, C, H, W)
            return Y, Y2

            # feature_E =
            # print(11)
            # return x_raw, x_raw

        # return Y
