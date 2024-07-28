import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from einops import rearrange


class UNet(nn.Module):
    def __init__(self, il=5, ol=5):
        super(UNet, self).__init__()
        self.ol = ol
        self.embed = DC(il, 16)
        enc = [DC(16, 32), DC(32, 64), DC(64, 128)]
        dec = [DC(64, 64), DC(32, 32), DC(16, 16)]
        # ds layers
        ds = [nn.MaxPool2d((2, 2)), nn.MaxPool2d((2, 2)), nn.MaxPool2d((2, 2))]
        # us
        us = [TC(128, 64), TC(64, 32), TC(32, 16)]

        self.enc = nn.ModuleList(enc)
        self.dec = nn.ModuleList(dec)
        self.ds = nn.ModuleList(ds)
        self.us = nn.ModuleList(us)
        # out
        self.out = DC(16, ol)

    def forward(self, x):
        # but actually get (b, t/l, ,c, h, w), so we rearrange
        """
        batch_size, seq_len, channel, height, width = x.size()
        x = rearrange(x, 'b l c h w -> (b l) c h w')
        x = F.leaky_relu(self.deconv1(x), negative_slope=0.1, inplace=True)
        x = F.leaky_relu(self.deconv2(x), negative_slope=0.1, inplace=True)
        x = t.sigmoid(self.conv(x))
        x = rearrange(F.interpolate(x, [int(height * 8), int(width * 8)], mode='bilinear'), '(b l) c h w -> b l c h w',
                      b=batch_size, l=seq_len)
        """
        batch_size, seq_len, channel, height, width = x.size()
        x = rearrange(x, 'b l c h w -> (b c) l h w')
        # x (bs, t, h, w)
        x_ = []
        x = self.embed(x)
        x_.append(x)
        for i in range(len(self.enc)):
            x = self.ds[i](x)
            x = self.enc[i](x)
            if i < len(self.enc)-1:
                x_.append(x)

        x_ = x_[::-1]
        for i in range(len(self.dec)):
            x = self.us[i](x)
            x = x + x_[i]
            x = self.dec[i](x)

        x = self.out(x)

        # x = rearrange(x, '(b c) l h w -> b l c h w',
        #               b=batch_size, c=1)
        x = rearrange(x, '(b c) l h w -> b l c h w',
                      b=batch_size, c=channel)
        # x = rearrange(x, 'b l c h w -> (b c) l h w')
        return x


class DC(nn.Sequential):
    def __init__(self, in_channels, out_channels, k=(3, 3), s=(1, 1), p=(1, 1)):
        super(DC, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=k, stride=s, padding=p, bias=False),
            # nn.GroupNorm(num_groups=1, num_channels=out_channels, eps=1e-6, affine=True, ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=k, stride=s, padding=p, bias=False),
            # nn.GroupNorm(num_groups=1, num_channels=out_channels, eps=1e-6, affine=True, ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class TC(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(TC, self).__init__(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            # nn.GroupNorm(num_groups=1, num_channels=out_channels, eps=1e-6, affine=True, ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.GroupNorm(num_groups=1, num_channels=out_channels, eps=1e-6, affine=True, ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
