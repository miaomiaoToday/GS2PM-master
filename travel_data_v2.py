"""Travel all datasets and render all images with inputs and outputs.
As the training dataset is shuffled, so the shuffle in this page is disabled.
Both training data and testing data is travelled and saved in 'travel_results'."""
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
from torch.utils.tensorboard import SummaryWriter
# additonally
from utils.loss_functions import weighted_l2_loss_radar
import sys
import numpy as np

in_len = 0
out_len = 0
# if configs.custom_len:
#     in_len = configs.in_len
#     out_len = configs.out_len


def travel():
    print("Time: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + " Travelling mode is a go. ")
    # load in_len and out_len & save path
    in_len, out_len = dataset.get_len(configs)
    log_dir = os.path.join(configs.travel_imgs_save_dir, configs.dataset_type)
    if configs.dataset_type == 'pam':
        pass
    # load dataloader
    train_dataloader, valid_dataloader, test_dataloader = dataset.load_data(configs)

    print("Time: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ' Load dataset successfully...')

    # travel data
    device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
    # for train
    len_train = len(train_dataloader)
    for iter, data in enumerate(train_dataloader):
        # Address data from dataloader
        if configs.dataset_type in ['hko7', 'shanghai2020', 'sevir-vil', 'sevir-vis', 'pam']:
            input = data[:, 0:in_len]
            ground_truth = data[:, in_len:(in_len + out_len)]
        else:
            input = data[0]
            ground_truth = data[1]
        if configs.use_gpu:
            input = input.to(device)
            ground_truth = ground_truth.to(device)

        utils.save_travel_imgs(os.path.join(log_dir, 'train'), iter, input, ground_truth,
                               configs.dataset_type, save_mode=configs.save_mode)
        print('Train iter: {}/{}: saved.'.format(iter, len_train))
    # for val
    if valid_dataloader is not None:
        len_valid = len(valid_dataloader)
        for iter, data in enumerate(valid_dataloader):
            # Address data from dataloader
            if configs.dataset_type in ['hko7', 'shanghai2020', 'sevir-vil', 'sevir-vis', 'pam']:
                input = data[:, 0:in_len]
                ground_truth = data[:, in_len:(in_len + out_len)]
            else:
                input = data[0]
                ground_truth = data[1]
            if configs.use_gpu:
                input = input.to(device)
                ground_truth = ground_truth.to(device)

            utils.save_travel_imgs(os.path.join(log_dir, 'valid'), iter, input, ground_truth,
                                   configs.dataset_type, save_mode=configs.save_mode)
            print('Valid iter: {}/{}: saved.'.format(iter, len_valid))
    # for test
    len_test = len(test_dataloader)
    for iter, data in enumerate(test_dataloader):
        # Address data from dataloader
        if configs.dataset_type in ['hko7', 'shanghai2020', 'sevir-vil', 'sevir-vis', 'pam']:
            input = data[:, 0:in_len]
            ground_truth = data[:, in_len:(in_len + out_len)]
        else:
            input = data[0]
            ground_truth = data[1]
        if configs.use_gpu:
            input = input.to(device)
            ground_truth = ground_truth.to(device)

        utils.save_travel_imgs(os.path.join(log_dir, 'test'), iter, input, ground_truth,
                               configs.dataset_type, save_mode=configs.save_mode)
        print('Test iter: {}/{}: saved.'.format(iter, len_test))


if __name__ == '__main__':
    if configs.mode == 'travel':
        travel()
    else:
        print('set the correct mode. hahahah')
