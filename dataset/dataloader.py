import os
import torch as t
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from configs import configs
import os
import utils
import models
import dataset
import torch as t
from torch import nn
from configs import configs
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader


def get_len(configs):
    """Obtain the input length and output length, provided the official length here.
    For 'shanghai2020' & 'taxibj', the settings are not able to be changed.
    The custom length could be setting at others."""
    len_official_dict = {
        'moving_mnist': [10, 10],
        'human': [4, 4],
        'kth': [10, 20],
        'bair': [4, 12],
        'kitticaltech': [12, 1],
        'taxibj': [4, 4],
        'hko7': [5, 15],
        'shanghai2020': [10, 10],
        'sevir-vil': [13, 18],
        'sevir-vis': [13, 18],
        'pam': [13, 18],
    }
    len_custom_dict = {
        'moving_mnist': [10, 10],
        'human': [10, 10],
        'kth': [10, 10],
        'bair': [10, 10],
        'kitticaltech': [10, 10],
        'kitticaltech_long': [100, 100],
        'taxibj': [4, 4],
        'hko7': [10, 10],
        'shanghai2020': [10, 10],
        'sevir-vil': [10, 10],
        'sevir-vis': [10, 10],
        'pam': [10, 10],
    }
    if configs.custom_len:
        [in_len, out_len] = len_custom_dict[configs.dataset_type]
    else:
        [in_len, out_len] = len_official_dict[configs.dataset_type]

    return in_len, out_len


def get_shape(configs):
    """Obtain the width and height for datasets. Disable the custom setting, all
    using the official setting."""
    shape_official_dict = {
        'moving_mnist': [64, 64, 1],
        'human': [128, 128, 3],
        'kth': [128, 128, 1],
        'bair': [64, 64, 3],
        'kitticaltech': [128, 160, 3],
        'kitticaltech_long': [128, 128, 3],
        'taxibj': [32, 32, 2],
        'hko7': [128, 128, 1],
        'shanghai2020': [128, 128, 1],
        'sevir-vil': [128, 128, 1],
        'sevir-vis': [128, 128, 1],
        'pam': [128, 128, 3],
        # 'sevir-vis': [768, 768, 1],
    }
    if configs.custom_shape == True:
        print('Warning! The shape is not supported with the self-defined way.')
    else:
        configs.img_width, configs.img_height, channel_num = shape_official_dict[configs.dataset_type]

    return configs.img_width, configs.img_height, channel_num


def get_dataloader(configs, in_len, out_len, img_width=None, img_height=None):
    """Obtain all dataloaders: train, val, test.
    Suggested that not all dataloaders stored, for memory saving. For faster training, load all still."""
    train_dataloader, valid_dataloader, test_dataloader = None, None, None
    if configs.dataset_type == 'hko7':
        train_dataset = dataset.HKO_7_Dataset(configs.dataset_root, seq_len=in_len + out_len, seq_interval=5,
                                              train=True, test=False, nonzero_points_threshold=6500)
        train_dataloader = DataLoader(train_dataset, batch_size=configs.batch_size, shuffle=True,
                                      num_workers=configs.num_workers, drop_last=True)
        valid_dataset = dataset.HKO_7_Dataset(configs.dataset_root, seq_len=in_len + out_len, seq_interval=5,
                                              train=False, test=False, nonzero_points_threshold=None)
        valid_dataloader = DataLoader(valid_dataset, configs.test_batch_size, shuffle=False,
                                      num_workers=configs.num_workers)
        test_dataset = dataset.HKO_7_Dataset(configs.dataset_root, seq_len=in_len + out_len, seq_interval=5,
                                             train=False, test=True, nonzero_points_threshold=None)
        test_dataloader = DataLoader(test_dataset, configs.test_batch_size, shuffle=False,
                                     num_workers=configs.num_workers)
    elif configs.dataset_type == 'shanghai2020':
        train_dataset = dataset.Shanghai_2020_Dataset(configs.dataset_root, seq_len=in_len + out_len, seq_interval=5,
                                                      train=True, test=False, nonzero_points_threshold=6500)
        train_dataloader = DataLoader(train_dataset, configs.batch_size, shuffle=True,
                                      num_workers=configs.num_workers, drop_last=True)
        valid_dataset = dataset.Shanghai_2020_Dataset(configs.dataset_root, seq_len=in_len + out_len, seq_interval=5,
                                                      train=False, test=False, nonzero_points_threshold=None)
        valid_dataloader = DataLoader(valid_dataset, configs.test_batch_size, shuffle=False,
                                      num_workers=configs.num_workers)
        test_dataset = dataset.Shanghai_2020_Dataset(configs.dataset_root, seq_len=in_len + out_len, seq_interval=5,
                                                     train=False, test=True, nonzero_points_threshold=None)
        test_dataloader = DataLoader(test_dataset, configs.test_batch_size, shuffle=False,
                                     num_workers=configs.num_workers)
    elif configs.dataset_type == 'bair':
        train_dataloader, valid_dataloader, test_dataloader = \
            dataset.load_data_bair(batch_size=configs.batch_size, val_batch_size=configs.test_batch_size,
                                   data_root=configs.dataset_root, num_workers=configs.num_workers,
                                   pre_seq_length=in_len, aft_seq_length=out_len, in_shape=[2, 3, 64, 64])
    elif configs.dataset_type == 'human':
        train_dataloader, valid_dataloader, test_dataloader = \
            dataset.load_data_human(batch_size=configs.batch_size, val_batch_size=configs.test_batch_size,
                                    data_root=configs.dataset_root, num_workers=configs.num_workers,
                                    pre_seq_length=in_len, aft_seq_length=out_len, in_shape=[4, 3, configs.img_width, configs.img_height],
                                    use_prefetcher=True, distributed=True)
    elif configs.dataset_type == 'kitticaltech':
        train_dataloader, valid_dataloader, test_dataloader = \
            dataset.load_data_kitticaltech(batch_size=configs.batch_size, val_batch_size=configs.test_batch_size,
                                           data_root=configs.dataset_root, num_workers=configs.num_workers,
                                           pre_seq_length=in_len, aft_seq_length=out_len, in_shape=[10, 3, 128, 160])
    elif configs.dataset_type == 'kth':
        train_dataloader, valid_dataloader, test_dataloader = \
            dataset.load_data_kth(batch_size=configs.batch_size, val_batch_size=configs.test_batch_size,
                                  data_root=configs.dataset_root, num_workers=configs.num_workers,
                                  pre_seq_length=in_len, aft_seq_length=out_len, in_shape=[10, 1, 128, 128])
    elif configs.dataset_type == 'moving_mnist':
        train_dataloader, valid_dataloader, test_dataloader = \
            dataset.load_data_moving_mnist(batch_size=configs.batch_size, val_batch_size=configs.test_batch_size,
                                           data_root=configs.dataset_root, num_workers=configs.num_workers,
                                           data_name='mnist',
                                           pre_seq_length=in_len, aft_seq_length=out_len, in_shape=[10, 1, 64, 64],
                                           distributed=True, use_prefetcher=False)
    elif configs.dataset_type == 'taxibj':
        train_dataloader, valid_dataloader, test_dataloader = \
            dataset.load_data_taxibj(batch_size=configs.batch_size, val_batch_size=configs.test_batch_size,
                                     data_root=configs.dataset_root, num_workers=configs.num_workers,
                                     pre_seq_length=in_len, aft_seq_length=out_len)
    elif configs.dataset_type == 'sevir-vil':
        train_dataset = dataset.SEVIR_challenge(configs.dataset_root, seq_len=in_len + out_len, seq_interval=5,
                                                train=True, test=False, task='vil')
        train_dataloader = DataLoader(train_dataset, configs.batch_size, shuffle=True,
                                      num_workers=configs.num_workers, drop_last=True)
        valid_dataset = dataset.SEVIR_challenge(configs.dataset_root, seq_len=in_len + out_len, seq_interval=5,
                                                train=False, test=False, task='vil')
        valid_dataloader = DataLoader(valid_dataset, configs.test_batch_size, shuffle=False,
                                      num_workers=configs.num_workers)
        test_dataset = dataset.SEVIR_challenge(configs.dataset_root, seq_len=in_len + out_len, seq_interval=5,
                                               train=False, test=True, task='vil')
        test_dataloader = DataLoader(test_dataset, configs.test_batch_size, shuffle=False,
                                     num_workers=configs.num_workers)
    elif configs.dataset_type == 'sevir-vis':
        train_dataset = dataset.SEVIR_challenge(configs.dataset_root, seq_len=in_len + out_len, seq_interval=5,
                                                train=True, test=False, task='vis')
        train_dataloader = DataLoader(train_dataset, configs.batch_size, shuffle=True,
                                      num_workers=configs.num_workers, drop_last=True)
        valid_dataset = dataset.SEVIR_challenge(configs.dataset_root, seq_len=in_len + out_len, seq_interval=5,
                                                train=False, test=False, task='vis')
        valid_dataloader = DataLoader(valid_dataset, configs.test_batch_size, shuffle=False,
                                      num_workers=configs.num_workers)
        test_dataset = dataset.SEVIR_challenge(configs.dataset_root, seq_len=in_len + out_len, seq_interval=5,
                                               train=False, test=True, task='vis')
        test_dataloader = DataLoader(test_dataset, configs.test_batch_size, shuffle=False,
                                     num_workers=configs.num_workers)
    elif configs.dataset_type == 'pam':
        train_dataset = dataset.PAM_TrainDataset(configs.dataset_root, seq_len=in_len + out_len)
        train_dataloader = DataLoader(train_dataset, configs.batch_size, shuffle=True,
                                      num_workers=configs.num_workers, drop_last=True)
        valid_dataloader = None
        test_dataloader = dataset.PAM_TestDataset(configs.dataset_root, seq_len=in_len + out_len)
        test_dataloader = DataLoader(test_dataloader, configs.test_batch_size, shuffle=False,
                                     num_workers=configs.num_workers)
    elif configs.dataset_type == 'kitticaltech_long':
        train_dataset = dataset.KitticaltechLong_TrainDataset(configs.dataset_root, seq_len=in_len + out_len)
        train_dataloader = DataLoader(train_dataset, configs.batch_size, shuffle=True,
                                      num_workers=configs.num_workers, drop_last=True)
        valid_dataloader = None
        test_dataloader = dataset.KitticaltechLong_TestDataset(configs.dataset_root, seq_len=in_len + out_len)
        test_dataloader = DataLoader(test_dataloader, configs.test_batch_size, shuffle=False,
                                     num_workers=configs.num_workers)

    return train_dataloader, valid_dataloader, test_dataloader


# Create dataloader
def load_data(configs):
    # get sequence length & save path
    in_len, out_len = get_len(configs)
    log_dir = os.path.join(configs.travel_imgs_save_dir, configs.dataset_type)

    # load dataloader
    train_dataloader, valid_dataloader, test_dataloader = get_dataloader(configs, in_len, out_len)

    return train_dataloader, valid_dataloader, test_dataloader


if __name__ == '__main__':
    print('test dataloaders')
    configs.dataset_root = '../data'
    train_dataloader, valid_dataloader, test_dataloader = load_data(configs)

    for iter in enumerate(train_dataloader):
        print('Train ok')
        break
    if valid_dataloader is not None:
        for iter in enumerate(valid_dataloader):
            print('Valid ok')
            break
    for iter in enumerate(test_dataloader):
        print('Test ok')
        break
