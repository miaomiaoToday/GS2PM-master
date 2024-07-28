import os
import torch as t
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from configs import configs
import natsort


class PAM_TrainDataset(Dataset):
    def __init__(self, root, seq_len=20, transforms=None):
        self.root = os.path.join(root, r'pam\train')
        self.seq_len = seq_len

        # read all dirs according to different domains.
        # domains = ['bair', 'hko7', 'human', 'kitticaltech', 'kth', 'moving_mnist', 'sevir-vil', 'sevir-vis', 'shanghai2020', 'taxibj']
        domains = ['bair', 'hko7', 'human', 'kitticaltech', 'kth', 'moving_mnist', 'sevir-vil', 'sevir-vis', 'shanghai2020']

        # check if finetune
        if configs.model == 'MS2Pv3_tune':
            domains = [configs.domain]

        self.data_paths = []
        for domain in domains:
            domain_path = os.path.join(self.root, domain)
            example_paths = os.listdir(domain_path)
            for example_path in example_paths:
                example_path = os.path.join(domain_path, example_path)
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
        for i in range(1, 11):
            img_name = os.path.join(example_dir, 'input' + str(i) + '.png')
            example_img = Image.open(img_name)
            example_img = example_img.convert("RGB")
            example_img = example_img.resize((configs.img_width, configs.img_height), Image.BILINEAR)
            example_img = self.transforms(example_img)
            example_imgs.append(example_img)
        # ground-truth
        for i in range(1, 11):
            img_name = os.path.join(example_dir, 'ground_truth' + str(i) + '.png')
            example_img = Image.open(img_name)
            example_img = example_img.convert("RGB")
            example_img = example_img.resize((configs.img_width, configs.img_height), Image.BILINEAR)
            example_img = self.transforms(example_img)
            example_imgs.append(example_img)

        example_imgs = t.stack(example_imgs, dim=0)
        return example_imgs
        # pass
        # example_path_aft = self.examples_path[item].split('\\')[-3:]
        # example_path_aft = os.path.join(example_path_aft[0], example_path_aft[1], example_path_aft[2])
        # example_path = os.path.join(self.root, example_path_aft)
        # example_index = example_path.split('\\')[-1]
        # example_imgs = []
        # f_inputs = open(os.path.join(example_path, example_index + '-inputs-train.txt'), 'r')
        # f_targets = open(os.path.join(example_path, example_index + '-targets-train.txt'), 'r')
        # for line in f_inputs.readlines():
        #     example_img_path = os.path.join(self.root, 'train', 'data', line.split('\n')[0])
        #     example_img = Image.open(example_img_path)
        #     # example_img = example_img.resize((256, 256), Image.BILINEAR)
        #     example_img = example_img.resize((configs.img_width, configs.img_height), Image.BILINEAR)
        #     example_img = self.transforms(example_img)
        #     example_imgs.append(example_img)
        # for line in f_targets.readlines():
        #     example_img_path = os.path.join(self.root, 'train', 'data', line.split('\n')[0])
        #     example_img = Image.open(example_img_path)
        #     # example_img = example_img.resize((256, 256), Image.BILINEAR)
        #     example_img = example_img.resize((configs.img_width, configs.img_height), Image.BILINEAR)
        #     example_img = self.transforms(example_img)
        #     example_imgs.append(example_img)
        # example_imgs = t.stack(example_imgs, dim=0)
        # return example_imgs

    def __len__(self):
        return len(self.data_paths)


class PAM_TestDataset(Dataset):
    def __init__(self, root, seq_len=20, transforms=None):
        self.root = os.path.join(root, r'pam\test')
        self.seq_len = seq_len

        # read all dirs according to different domains.
        # domains = ['bair', 'hko7', 'human', 'kitticaltech', 'kth', 'moving_mnist', 'sevir-vil', 'sevir-vis', 'shanghai2020', 'taxibj']
        domains = ['bair', 'hko7', 'human', 'kitticaltech', 'kth', 'moving_mnist', 'sevir-vil', 'sevir-vis', 'shanghai2020']
        self.domain = configs.domain
        assert self.domain in domains, f"{self.domain} not in list"

        self.data_paths = []
        # for domain in domains:
        domain_path = os.path.join(self.root, self.domain)
        example_paths = os.listdir(domain_path)
        example_paths = natsort.natsorted(example_paths)

        for example_path in example_paths:
            # print(example_path)
            example_path = os.path.join(domain_path, example_path)
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
        for i in range(1, 11):
            img_name = os.path.join(example_dir, 'input' + str(i) + '.png')
            example_img = Image.open(img_name)
            example_img = example_img.convert("RGB")
            example_img = example_img.resize((configs.img_width, configs.img_height), Image.BILINEAR)
            example_img = self.transforms(example_img)
            example_imgs.append(example_img)
        # ground-truth
        for i in range(1, 11):
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
