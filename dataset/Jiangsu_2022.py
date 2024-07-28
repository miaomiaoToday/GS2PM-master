import os
import csv
import torch as t
import numpy as np
from imageio import imread
from torch.utils.data import Dataset
from torchvision import transforms as T
# from torch.nn.functional import interpolate
from skimage.transform import resize
# import Image
# import scipy
from PIL import Image
from configs import configs


class Jiangsu_2022(Dataset):
    def __init__(self, daset_root, transforms=None, train=True):
        self.daset_root = daset_root
        self.train = train
        self.folder = 'Train' if self.train else 'TestA'
        self.cases = []
        if self.train:
            f = csv.reader(open(os.path.join(self.daset_root, 'Train.csv'), 'r', encoding='utf-8-sig'))
        else:
            f = csv.reader(open(os.path.join(self.daset_root, 'TestA_valid.csv'), 'r', encoding='utf-8-sig'))
        for row in f:
            self.cases.append(row)

        if transforms is None:
            self.transforms = T.Compose([T.ToTensor()])
        else:
            self.transforms = transforms

    # output: img_seq (seq_len=40, channels=1, height=480, width=560) [0,1]
    def __getitem__(self, item):
        img_seq = []
        for i in range(len(self.cases[item])):
            img_index = self.cases[item][i]
            img_path = os.path.join(self.daset_root, self.folder, 'Radar', 'radar_' + img_index)
            # img = self.transforms(np.array(imread(img_path)))
            img = imread(img_path)
            # img = np.array(Image.fromarray(img).resize((120, 120)))
            img = np.array(Image.fromarray(img).resize((configs.img_width, configs.img_height)))
            img = self.transforms(Image.fromarray(img))

            img_seq.append(img)
        img_seq = t.stack(img_seq, dim=0)
        return img_seq

    def __len__(self):
        return len(self.cases)


class Jiangsu_2022_Test(Dataset):
    def __init__(self, daset_root, transforms=None):
        self.daset_root = daset_root
        self.folder = 'TestB1'
        self.cases = os.listdir(os.path.join(self.daset_root, self.folder, 'Radar'))
        """for sorting the files on both win and lin"""
        # self.cases.sort(key=lambda x: int(x))
        self.cases.sort()

        if transforms is None:
            self.transforms = T.Compose([T.ToTensor()])
        else:
            self.transforms = transforms

    # output: img_seq (seq_len=20, channels=1, height=480, width=560) [0,1]
    def __getitem__(self, item):
        img_seq = []
        """for sorting files on both..."""
        images = os.listdir(os.path.join(self.daset_root, self.folder, 'Radar', self.cases[item]))
        images.sort()
        imgs_paths = [os.path.join(self.daset_root, self.folder, 'Radar', self.cases[item], img_path) for img_path in
                      images]
        for i in range(len(imgs_paths)):
            img_path = imgs_paths[i]
            # img = self.transforms(np.array(imread(img_path)))
            img = imread(img_path)
            # img = np.array(Image.fromarray(img).resize((120, 120)))
            img = np.array(Image.fromarray(img).resize((configs.img_width, configs.img_height)))
            img = self.transforms(img)

            img_seq.append(img)
        img_seq = t.stack(img_seq, dim=0)
        return img_seq

    def __len__(self):
        return len(self.cases)


class Jiangsu_2022_Test_hq(Dataset):
    def __init__(self, daset_root, transforms=None):
        self.daset_root = daset_root
        self.folder = 'TestB1'
        self.cases = os.listdir(os.path.join(self.daset_root, self.folder, 'Radar'))
        """for sorting the files on both win and lin"""
        # self.cases.sort(key=lambda x: int(x))
        self.cases.sort()

        if transforms is None:
            self.transforms = T.Compose([T.ToTensor()])
        else:
            self.transforms = transforms

    # output: img_seq (seq_len=20, channels=1, height=480, width=560) [0,1]
    def __getitem__(self, item):
        img_seq = []
        """for sorting files on both..."""
        images = os.listdir(os.path.join(self.daset_root, self.folder, 'Radar', self.cases[item]))
        images.sort()
        imgs_paths = [os.path.join(self.daset_root, self.folder, 'Radar', self.cases[item], img_path) for img_path in
                      images]
        for i in range(len(imgs_paths)):
            img_path = imgs_paths[i]
            # img = self.transforms(np.array(imread(img_path)))
            img = imread(img_path)
            img = np.array(Image.fromarray(img).resize((120, 120)))
            img = self.transforms(img)

            img_seq.append(img)
        img_seq = t.stack(img_seq, dim=0)
        return img_seq

    def __len__(self):
        return len(self.cases)
