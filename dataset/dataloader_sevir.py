import os
import torch as t
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from configs import configs
import h5py
import pandas as pd
import numpy as np


DATA_PATH    = configs.dataset_root + r'\sevir\data'
CATALOG_PATH = configs.dataset_root + r'\sevir\CATALOG.csv'
os.environ["HDF5_USE_FILE_LOCKING"]='FALSE'
import matplotlib.pyplot as plt
import pandas as pd
import math


def read_data(sample_event, img_type, data_path=DATA_PATH):
    """
    Reads single SEVIR event for a given image type.

    Parameters
    ----------
    sample_event   pd.DataFrame
        SEVIR catalog rows matching a single ID
    img_type   str
        SEVIR image type
    data_path  str
        Location of SEVIR data

    Returns
    -------
    np.array
       LxLx49 tensor containing event data
    """
    fn = sample_event[sample_event.img_type == img_type].squeeze().file_name
    fi = sample_event[sample_event.img_type == img_type].squeeze().file_index
    #print('data_path:{}'.format(data_path))
    #print('fn: {}'.format(fn))
    #print('fi: {}'.format(fi))
    with h5py.File(data_path + '/' + fn, 'r') as hf:

        data = hf[img_type][fi]
    return data


def lght_to_grid(data):
    """
    Converts SEVIR lightning data stored in Nx5 matrix to an LxLx49 tensor representing
    flash counts per pixel per frame

    Parameters
    ----------
    data  np.array
       SEVIR lightning event (Nx5 matrix)

    Returns
    -------
    np.array
       LxLx49 tensor containing pixel counts
    """
    FRAME_TIMES = np.arange(-120.0, 125.0, 5) * 60  # in seconds
    out_size = (48, 48, len(FRAME_TIMES))
    if data.shape[0] == 0:
        return np.zeros(out_size, dtype=np.float32)

    # filter out points outside the grid
    x, y = data[:, 3], data[:, 4]
    m = np.logical_and.reduce([x >= 0, x < out_size[0], y >= 0, y < out_size[1]])
    data = data[m, :]
    if data.shape[0] == 0:
        return np.zeros(out_size, dtype=np.float32)

    # Filter/separate times
    # compute z coodinate based on bin locaiton times
    t = data[:, 0]
    z = np.digitize(t, FRAME_TIMES) - 1
    z[z == -1] = 0  # special case:  frame 0 uses lght from frame 1

    x = data[:, 3].astype(np.int64)
    y = data[:, 4].astype(np.int64)

    k = np.ravel_multi_index(np.array([y, x, z]), out_size)
    n = np.bincount(k, minlength=np.prod(out_size))
    return np.reshape(n, out_size).astype(np.float32)


def read_lght_data(sample_event, data_path=DATA_PATH):
    """
    Reads lght data from SEVIR and maps flash counts onto a grid

    Parameters
    ----------
    sample_event   pd.DataFrame
        SEVIR catalog rows matching a single ID
    data_path  str
        Location of SEVIR data

    Returns
    -------
    np.array
       LxLx49 tensor containing pixel counts for selected event

    """
    fn = sample_event[sample_event.img_type == 'lght'].squeeze().file_name
    id = sample_event[sample_event.img_type == 'lght'].squeeze().id
    with h5py.File(data_path + '/' + fn, 'r') as hf:
        data = hf[id][:]
    return lght_to_grid(data)


class SEVIR_challenge(Dataset):
    def __init__(self, root, seq_len, seq_interval=5, transforms=None, train=True, test=False, task='vil'):
        self.root = root
        self.train = train
        self.test = test
        self.seq_len = seq_len
        self.seq_interval = seq_interval
        self.rainy_days_imgs_indexs = []
        self.X_VIL = []     # used for all kinds of situations. for get item
        self.X_LGHT = []    # used for all kinds of situations. for get item
        self.task = task

        """sevir"""
        # Read catalog
        catalog = pd.read_csv(CATALOG_PATH, parse_dates=['time_utc'], low_memory=False)

        # Desired image types
        # img_types = set(['ir069', 'ir107', 'vil', 'lght'])
        if self.task == 'vil':
            img_types = set(['vis', 'ir069', 'ir107', 'vil', 'lght'])
        elif self.task == 'vis':
            img_types = set(['vis', 'ir069', 'ir107', 'vil', 'lght'])
        else:
            print('not signed')

        # Group by event id, and filter to only events that have all desired img_types
        events = catalog.groupby('id').filter(lambda x: img_types.issubset(set(x['img_type']))).groupby('id')
        event_ids = list(events.groups.keys())
        print('Found %d events matching' % len(event_ids), img_types)

        # Grab a sample event and view catalog entries
        # sample_event = events.get_group(event_ids[-1])
        # print('Sample Event:', event_ids[-1])
        # print(sample_event)
        self.events = events
        self.event_ids = event_ids
        self.train_ids = []
        self.val_ids = []
        self.test_ids = []
        self.cur_ids = []
        if self.train:
            self.cur_ids = [i for i in range(math.floor(len(event_ids) * 0.0),
                                              math.floor(len(event_ids) * 0.6))] #[i for i in range(0, math.floor(len(event_ids)*0.1))]

        elif self.test:
            self.cur_ids = [i for i in range(math.floor(len(event_ids) * 0.8),
                                              math.floor(len(event_ids) * 1.0))]
        else:
            self.cur_ids = [i for i in range(math.floor(len(event_ids) * 0.6),
                                              math.floor(len(event_ids) * 0.8))]

        if transforms is None:
            self.transforms = T.Compose([T.ToTensor()])
        else:
            self.transforms = transforms

    # output: rainy_day_sequence_imgs [seq_len, channels, height, width]
    def __getitem__(self, item):
        cid = self.cur_ids[item]
        sample_event = self.events.get_group(self.event_ids[cid])
        # print('Sample Event:', self.event_ids[cid])
        # print(sample_event)
        try:
            if self.task == 'vis':
                vis = read_data(sample_event, 'vis')
            elif self.task == 'vil':
                vil = read_data(sample_event, 'vil')
        except:
            sample_event = self.events.get_group(self.event_ids[-1])
            if self.task == 'vis':
                vis = read_data(sample_event, 'vis')
            elif self.task == 'vil':
                vil = read_data(sample_event, 'vil')

        vis_seqs = []
        vil_seqs = []
        for i in range(self.seq_len):
            if self.task == 'vis':      # encoded
                each_vis = vis[:, :, i] * 0.0001
                each_vis = self.transforms(each_vis)
                resize = T.Resize([configs.img_width, configs.img_height])
                each_vis = resize(each_vis)
                each_vis = each_vis.float()
                vis_seqs.append(each_vis)
            elif self.task == 'vil':
                each_vil = vil[:, :, i]
                each_vil = each_vil.astype(np.uint8)
                each_vil = Image.fromarray(each_vil)
                each_vil = each_vil.resize((configs.img_width, configs.img_height), Image.BILINEAR)
                each_vil = self.transforms(each_vil)
                vil_seqs.append(each_vil)

        out = None
        if self.task == 'vis':
            vis_seqs = t.stack(vis_seqs, dim=0)
            out = vis_seqs
        elif self.task == 'vil':
            vil_seqs = t.stack(vil_seqs, dim=0)
            out = vil_seqs

        # return ir069_seqs, ir107_seqs, vil_seqs, lght_seqs
        return out


    def __len__(self):
        # print(len(self.X_VIL))
        return len(self.cur_ids)





