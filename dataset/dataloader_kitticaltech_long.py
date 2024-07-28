import os
import torch as t
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from configs import configs
import natsort
import math


class KitticaltechLong_TrainDataset(Dataset):
    def __init__(self, root, seq_len=200, transforms=None):
        self.root = os.path.join(root, r'kitticaltech_long\train')
        self.seq_len = seq_len

        self.data_paths = []
        example_paths = os.listdir(self.root)
        for example_path in example_paths:
            example_path = os.path.join(self.root, example_path)
            self.data_paths.append(example_path)

        if transforms is None:
            self.transforms = T.Compose([T.ToTensor()])
        else:
            self.transforms = transforms

        # print(len(self.data_paths))       670098

    # output: example_imgs [seq_len, channels, height, width]
    def __getitem__(self, item):
        example_dir = self.data_paths[item]
        # img_names = os.listdir(example_dir)
        # print(item, img_names)

        # for i in range(1, 1 + configs.in_len):
        example_imgs = []
        # input
        for i in range(1, 101):
            img_name = os.path.join(example_dir, 'input' + str(i) + '.png')
            example_img = Image.open(img_name)
            example_img = example_img.convert("RGB")
            example_img = example_img.resize((configs.img_width, configs.img_height), Image.BILINEAR)
            example_img = self.transforms(example_img)
            example_imgs.append(example_img)
        # ground-truth
        for i in range(1, 101):
            img_name = os.path.join(example_dir, 'ground_truth' + str(i) + '.png')
            example_img = Image.open(img_name)
            example_img = example_img.convert("RGB")
            example_img = example_img.resize((configs.img_width, configs.img_height), Image.BILINEAR)
            example_img = self.transforms(example_img)
            example_imgs.append(example_img)

        example_imgs = t.stack(example_imgs, dim=0)
        return example_imgs

    def __len__(self):
        return len(self.data_paths)


class KitticaltechLong_TestDataset(Dataset):
    def __init__(self, root, seq_len=200, transforms=None):
        self.root = os.path.join(root, r'kitticaltech_long\test')
        self.seq_len = seq_len

        self.data_paths = []
        example_paths = os.listdir(self.root)
        example_paths = natsort.natsorted(example_paths)
        example_paths = example_paths[-math.floor(len(example_paths) * 0.1):]

        for example_path in example_paths:
            # print(example_path)
            example_path = os.path.join(self.root, example_path)
            self.data_paths.append(example_path)

        if transforms is None:
            self.transforms = T.Compose([T.ToTensor()])
        else:
            self.transforms = transforms

        # print(len(self.data_paths))       670098

    # output: example_imgs [seq_len, channels, height, width]
    def __getitem__(self, item):
        example_dir = self.data_paths[item]
        # img_names = os.listdir(example_dir)
        # print(item, img_names)

        # for i in range(1, 1 + configs.in_len):
        example_imgs = []
        # input
        for i in range(1, 101):
            img_name = os.path.join(example_dir, 'input' + str(i) + '.png')
            example_img = Image.open(img_name)
            example_img = example_img.convert("RGB")
            example_img = example_img.resize((configs.img_width, configs.img_height), Image.BILINEAR)
            example_img = self.transforms(example_img)
            example_imgs.append(example_img)
        # ground-truth
        for i in range(1, 101):
            img_name = os.path.join(example_dir, 'ground_truth' + str(i) + '.png')
            try:
                example_img = Image.open(img_name)
                example_img = example_img.convert("RGB")
                example_img = example_img.resize((configs.img_width, configs.img_height), Image.BILINEAR)
                example_img = self.transforms(example_img)
                example_imgs.append(example_img)
            except:
                print('no image')

        example_imgs = t.stack(example_imgs, dim=0)
        return example_imgs

    def __len__(self):
        return len(self.data_paths)
