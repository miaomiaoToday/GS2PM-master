import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import pdb
import math


class Grid(object):
    def __init__(self, d1, d2, rotate=1, ratio=0.5, mode=0, prob=1.):
        self.d1 = d1
        self.d2 = d2
        self.rotate = rotate
        self.ratio = ratio
        self.mode = mode
        self.st_prob = self.prob = prob

    def set_prob(self, epoch, max_epoch):
        # self.prob = self.st_prob * min(1, epoch / max_epoch)
        self.prob = self.st_prob * 1
        # print(self.prob)

    def __call__(self, img):
        if np.random.rand() > self.prob:
            return img
        h = img.size(1)
        w = img.size(2)

        # 1.5 * h, 1.5 * w works fine with the squared images
        # But with rectangular input, the mask might not be able to recover back to the input image shape
        # A square mask with edge length equal to the diagnoal of the input image
        # will be able to cover all the image spot after the rotation. This is also the minimum square.
        hh = math.ceil((math.sqrt(h * h + w * w)))

        d = np.random.randint(self.d1, self.d2)
        # d = self.d1

        # maybe use ceil? but i guess no big difference
        rand_ratio = np.random.rand()

        if rand_ratio > self.ratio:
            temp_ratio = rand_ratio
        else:
            temp_ratio = self.ratio

        self.l = math.ceil(d * self.ratio)
        # self.l = math.ceil(d * temp_ratio)

        mask = np.ones((hh, hh), np.float32)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)
        for i in range(-1, hh // d + 1):
            s = d * i + st_h
            t = s + self.l
            s = max(min(s, hh), 0)
            t = max(min(t, hh), 0)
            mask[s:t, :] *= 0
        for i in range(-1, hh // d + 1):
            s = d * i + st_w
            t = s + self.l
            s = max(min(s, hh), 0)
            t = max(min(t, hh), 0)
            mask[:, s:t] *= 0
        r = np.random.randint(self.rotate)
        mask = Image.fromarray(np.uint8(mask))
        # mask = mask.rotate(r)
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        mask = mask[(hh - h) // 2:(hh - h) // 2 + h, (hh - w) // 2:(hh - w) // 2 + w]
        # print(mask)
        mask = torch.from_numpy(mask).float().cuda()
        if self.mode == 1:
            mask = 1 - mask

        mask = mask.expand_as(img)
        # mask = mask.cpu()   # zhushi for final trainging ,not do it for ablation s.
        img = img.cuda()    # zhishi for ablation s.
        img = img * mask

        return img


class GridMask(nn.Module):
    def __init__(self, d1, d2, rotate=1, ratio=0.5, mode=0, prob=1.):
        super(GridMask, self).__init__()
        self.rotate = rotate
        self.ratio = ratio
        self.mode = mode
        self.st_prob = prob
        self.grid = Grid(d1, d2, rotate, ratio, mode, prob)

    def set_prob(self, epoch, max_epoch):
        self.grid.set_prob(epoch, max_epoch)

    def forward(self, x):
        # if not self.training:
        #     return x
        # n, c, h, w = x.size()
        # y = []
        # for i in range(n):
        #     y.append(self.grid(x[i]))
        # y = torch.cat(y).view(n, c, h, w)
        # return y
        c, h, w = x.size()
        y = []
        y.append(self.grid(x))
        y = torch.cat(y).view(c, h, w)
        return y
        # return x
