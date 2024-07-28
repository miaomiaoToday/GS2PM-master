import os
from logging import getLogger
logger = getLogger()
import torch
import torchvision.transforms as transforms
import torchvision
from PIL import ImageFilter


class GaussianBlur(object):
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        if torch.bernoulli(torch.tensor(self.prob)) == 0:
            return img

        radius = self.radius_min + torch.rand(1) * (self.radius_max - self.radius_min)
        return img.filter(ImageFilter.GaussianBlur(radius=radius))


def make_transforms(
        rand_size=128,
        focal_size=96,
        rand_crop_scale=(0.3, 1.0),
        focal_crop_scale=(0.05, 0.3),
        rand_views=2,
        focal_views=10,
):
    logger.info('making training data transforms')

    ori = transforms.Compose([
        # transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False),
        transforms.ToTensor(),
    ])
    rand_eras = transforms.Compose([
        # transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False),
        transforms.RandomErasing(p=0.5),
        transforms.ToTensor(),
    ])
    rand_transform = transforms.Compose([
        transforms.RandomResizedCrop(rand_size, scale=rand_crop_scale),
        transforms.RandomHorizontalFlip(),
        # GaussianBlur(p=0.5),
        transforms.ToTensor(),
        # transforms.Normalize(
        #     (0.485, 0.456, 0.406),
        #     (0.229, 0.224, 0.225))
    ])
    focal_transform = transforms.Compose([
        transforms.RandomResizedCrop(focal_size, scale=focal_crop_scale),
        transforms.RandomHorizontalFlip(),
        # GaussianBlur(p=0.5),
        transforms.ToTensor(),
        # transforms.Normalize(
        #     (0.485, 0.456, 0.406),
        #     (0.229, 0.224, 0.225))
    ])
    # transform = MultiViewTransform(
    #     rand_transform=rand_transform,
    #     # focal_transform=focal_transform,
    #     rand_views=rand_views,
    #     focal_views=focal_views
    # )
    # transform = rand_transform
    transform = ori
    return transform


class MultiViewTransform(object):

    def __init__(
            self,
            rand_transform=None,
            focal_transform=None,
            rand_views=1,
            focal_views=1,
    ):
        self.rand_views = rand_views
        self.focal_views = focal_views
        self.rand_transform = rand_transform
        self.focal_transform = focal_transform

    def __call__(self, img):
        img_views = []

        # -- generate random views
        if self.rand_views > 0:
            img_views += [self.rand_transform(img) for i in range(self.rand_views)]

        # -- generate focal views
        if self.focal_views > 0:
            img_views += [self.focal_transform(img) for i in range(self.focal_views)]

        return img_views



