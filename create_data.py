"""
For creating the large scale of data for training our large model.
training data.
"""
import os
import utils
import dataset
import torch as t
from configs import configs
from datetime import datetime


def create_data():
    print("Time: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + " Creating data is a go. ")

    in_len, out_len = dataset.get_len(configs)
    log_dir = os.path.join(configs.ourdata_save_dir, configs.dataset_type)
    # load dataloader
    train_dataloader, _, test_dataloader = dataset.load_data(configs)
    print("Time: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ' Load dataset successfully...')

    # travel data
    device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
    # for train
    len_train = len(test_dataloader)
    for iter, data in enumerate(test_dataloader):
        # address memory slow # shuffle should be false, bair is set false, please set back to true
        # if iter <= 144614:
        #     print(iter)
        #     continue
        # Address data from dataloader
        if configs.dataset_type in ['hko7', 'shanghai2020', 'sevir-vil', 'sevir-vis']:
            input = data[:, 0:in_len]
            ground_truth = data[:, in_len:(in_len + out_len)]
        else:
            input = data[0]
            ground_truth = data[1]
        if configs.use_gpu:
            input = input.to(device)
            ground_truth = ground_truth.to(device)

        utils.save_travel_imgs(os.path.join(log_dir), iter, input, ground_truth,
                               configs.dataset_type, save_mode=configs.save_mode)
        print('Train iter: {}/{}: saved.'.format(iter, len_train))


if __name__ == '__main__':
    if configs.mode == 'create':
        create_data()
    else:
        print('set the correct mode. hahahah')
