"""This is the main function for experiments with Benchmark
Precipitation Nowcasting (BPN)."""
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
# from utils.convlstmnet import ConvLSTMNet
from utils.loss_functions import weighted_l2_loss_radar
import sys
from models.constrain_moments import K2M
import numpy as np

import model_dict

#
in_len = configs.in_len
out_len = configs.out_len


def ini_model_params(model, ini_mode='xavier'):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.Linear)):
            if ini_mode == 'xavier':
                nn.init.xavier_normal_(m.weight)
            elif ini_mode == 'orthogonal':
                nn.init.orthogonal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)


def test(test_dataloader=None, mode='weather', model_name=None, dBZ_threshold=10):
    print("Time: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "  Start Test")

    if model_name == None:
        model_name = configs.model_save_dir + '/' + configs.pretrained_model + '.pth'
    else:
        print('Testing model: {}'.format(model_name))

    """Load dataloader"""
    if test_dataloader == None:
        _, _, test_dataloader = dataset.load_data(configs)
    print("Time: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') +
          'Load dataset successfully...')

    # Model setting
    """Load in_len, out_len, shape"""
    in_len, out_len = dataset.get_len(configs)
    img_width, img_height, channel_num = dataset.get_shape(configs)
    log_dir = os.path.join(configs.test_imgs_save_dir, configs.dataset_type)
    if configs.dataset_type == 'pam':
        log_dir = os.path.join(log_dir, configs.domain)
    log_dir = os.path.join(log_dir, configs.pretrained_model)

    if configs.model == 'MS2Pv2' or configs.model == 'MS2Pv3' or configs.model == 'MS2Pv3_tune':
        channel_num = 3

    """Load model"""
    print("Time: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ' Model structure: \t {}'.format(configs.model))
    model = model_dict.load_model(configs, in_len, out_len, img_width, img_height, channel_num)

    model.load_state_dict(t.load(model_name))
    if configs.use_gpu:
        device = t.device('cuda:0')
        if len(configs.device_ids_eval) > 1:
            model = nn.DataParallel(model, device_ids=configs.device_ids_eval, dim=0)
            model.to(device)
        else:
            model.to(device)
    print("Time: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "  Load successfully {}".format(model_name))

    # indexes
    model.eval()
    if mode == 'weather':
        pod, far, csi, bias, hss, index, ssim, score = [], [], [], [], [], [], [], []
    elif mode == 'simple':
        mse, psnr, ssim, lpips = [], [], [], []

    with t.no_grad():
        l_t = len(test_dataloader.dataset)
        fq = round(l_t / 100)
        i = 0
        for iter, data in enumerate(test_dataloader):
            # if iter == 2:
            #     break
            if iter % fq == 0:
                i += 1
            # print(iter%fq, l_t)
            print("\r", end="")
            print("valid progress: {}%: ".format(i), "▋" * (i // 2), end="")

            # Address data from dataloader
            if configs.dataset_type in ['hko7', 'shanghai2020', 'sevir-vil', 'sevir-vis', 'pam']:
                input = data[:, 0:in_len]
                ground_truth = data[:, in_len:(in_len + out_len)]
            else:
                input = data[0]
                ground_truth = data[1]
            if configs.use_gpu:
                device = t.device('cuda:0')
                input = input.to(device)
                ground_truth = ground_truth.to(device)

            if configs.model == 'MS2Pv3' and input.size(2) == 1:
                # print(input.size())
                input = input.repeat(1, 1, 3, 1, 1)
                # print(input.size())
                ground_truth = ground_truth.repeat(1, 1, 3, 1, 1)
            if configs.model == 'MS2Pv3_tune' and input.size(2) == 1:
                # print(input.size())
                input = input.repeat(1, 1, 3, 1, 1)
                # print(input.size())
                ground_truth = ground_truth.repeat(1, 1, 3, 1, 1)
            # Prepare output (generating output on input) and ground-truth
            # confirm the performance
            # ground_truth = input
            if configs.model in ['ConvLSTMNet', 'PredRNN', 'PredRNN_patch', 'UNet', 'ConvLSTM', 'MIM', 'Eidetic3DLSTM',
                                 'MotionRNN', 'TrajGRU', 'PredRNNpp', 'LMC', 'SimVP', 'SimVPv2', 'MS2P', 'MS2Pv2',
                                 'MS2Pv3', 'MS2Pv3_tune']:
                try:
                    output, ground_truth = model_dict.model_forward(model, in_len, out_len, input, ground_truth,
                                                                    configs)
                except RuntimeError as e:
                    print('Runtime Error ' + str(e))

            elif configs.model == 'PhyDNet':
                for ei in range(in_len - 1):
                    encoder_output, encoder_hidden, _, _, _ = model(input[:, ei, :, :, :], (ei == 0))

                decoder_input = input[:, -1, :, :, :]  # first decoder input= last image of input sequence
                predictions = []

                for di in range(out_len):
                    decoder_output, decoder_hidden, output_image, _, _ = model(decoder_input, False, False)
                    decoder_input = output_image
                    predictions.append(output_image.cpu())
                    # predictions.append(output_image)
                    # ground_truth = t.cat([input[:, 1:in_len], ground_truth], dim=1)

                predictions = np.stack(predictions)  # (10, batch_size, 1, 64, 64)
                predictions = predictions.swapaxes(0, 1)  # (batch_size,10, 1, 64, 64)
                output = predictions
                ground_truth = ground_truth
                output = t.from_numpy(output)
                output = output.to(device)

            if configs.model == 'MS2Pv3':
                output = output[0]

            if mode == 'weather':
                pod_, far_, csi_, bias_, hss_, index_ = utils.crosstab_evaluate(output, ground_truth, dBZ_threshold, 70,
                                                                                dataset)
            elif mode == 'simple':
                # mse_, psnr_, lpips_ = utils.crosstab_evaluate_simple(output, ground_truth)
                try:
                    mse_, psnr_, lpips_ = utils.crosstab_evaluate_simple(output, ground_truth)
                except:
                    print('something wrong, if for kitticaltech, the len of in and out is not ok in 1st sample test.')
                    continue

            if mode == 'weather':
                pod.append(pod_.data)
                far.append(far_.data)
                csi.append(csi_.data)
                bias.append(bias_.data)
                hss.append(hss_.data)
                index.append(index_)
                ssim_ = utils.compute_ssim(output.cpu().numpy(), ground_truth.cpu().numpy())
                ssim.append(ssim_)
            elif mode == 'simple':
                mse.append(mse_)
                psnr.append(psnr_)
                lpips.append(lpips_)
                ssim_ = utils.compute_ssim(output.cpu().numpy(), ground_truth.cpu().numpy())
                ssim_ = ssim_.cpu().numpy()
                ssim.append(ssim_)
            # print(index_.size())

            # save seqs ?
            if configs.save_open:
                utils.save_test_imgs(log_dir, iter, input, output, ground_truth,
                                     configs.dataset_type, save_mode=configs.save_mode)
            # if iter >= 50:
            #     break
        # index = t.cat(index, dim=1)  # not appropriate for ours
        if mode == 'weather':
            index = t.cat(index, dim=0)
            data_num = index.numel()
            # the ground-truth sample which has no rainfall preddiction hits will not be included in calculation
            # cal_num = index.size()[1] - t.sum(index, dim=1) if eval_by_seq is True else data_num - t.sum(index)  # not apppri...
            # print(cal_num)
            cal_num = data_num - t.sum(index)
            # print(t.sum(index))

            pod = out_len * t.sum(t.cat(pod, dim=0), 0) / cal_num
            far = out_len * t.sum(t.cat(far, dim=0), 0) / cal_num
            csi = out_len * t.sum(t.cat(csi, dim=0), 0) / cal_num
            bias = out_len * t.sum(t.cat(bias, dim=0), 0) / cal_num
            hss = out_len * t.sum(t.cat(hss, dim=0), 0) / cal_num
            ssim = t.mean(t.cat(ssim, dim=0), 0)

            # pod_sum = t.sum(t.cat(pod, dim=0)) / cal_num
            # far_sum = t.sum(t.cat(far, dim=0)) / cal_num
            # csi_sum = t.sum(t.cat(csi, dim=0)) / cal_num
            # bias_sum = t.sum(t.cat(bias, dim=0)) / cal_num
            # hss_sum = t.sum(t.cat(hss, dim=0)) / cal_num
            # ssim_sum = t.mean(t.cat(ssim, dim=0))
        elif mode == 'simple':
            mse = np.average(mse, axis=0)
            psnr = np.average(psnr, axis=0)
            ssim = np.average(ssim, axis=0)
            lpips = np.average(lpips, axis=0)
            # pass

    model.train()
    if mode == 'weather':
        print('\n pod: {}\n far: {}\n csi: {}\n bias: {}\n hss: {}\n ssim: {}\n'.format(pod, far, csi, bias, hss, ssim))
        print('Time: ' + datetime.now().strftime(
            '%Y-%m-%d %H:%M:%S') + '  Test:\tPOD: {:.4f}, FAR: {:.4f}, CSI: {:.4f}, BIAS: {:.4f}, HSS: {:.4f}, SSIM: {:.4f}'
              .format(t.mean(pod), t.mean(far), t.mean(csi), t.mean(bias), t.mean(hss), t.mean(ssim)))
    elif mode == 'simple':
        print('mse: {}\n psnr: {}\n ssim: {}\n lpips: {}\n'.format(mse, psnr, ssim, lpips))
        print('Time: ' + datetime.now().strftime(
            '%Y-%m-%d %H:%M:%S') + '  Test:\tMSE: {:.4f}, PSNR: {:.4f}, SSIM: {:.4f}, LPIPS: {:.4f}'
              .format(np.average(mse), np.average(psnr), np.average(ssim), np.average(lpips)))

    # if not os.path.exists(configs.test_imgs_save_dir):
    #     os.makedirs(configs.test_imgs_save_dir)
    # if mode == 'weather':
    #     utils.save_test_results(configs.test_imgs_save_dir, pod, far, csi, bias, hss, ssim)
    # elif mode == 'simple':
    #     utils.save_test_results(configs.test_imgs_save_dir, mse, psnr, ssim, lpips)

    if mode == 'weather':
        return pod, far, csi, bias, hss, ssim
    elif mode == 'simple':
        return mse, psnr, ssim, lpips


def train():
    print("Time: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + " Train mode is a go. ")
    """Load in_len, out_len, shape"""
    in_len, out_len = dataset.get_len(configs)
    img_width, img_height, channel_num = dataset.get_shape(configs)

    """Load model"""
    print("Time: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ' Model structure: \t {}'.format(configs.model))
    model = model_dict.load_model(configs, in_len, out_len, img_width, img_height, channel_num)

    # Pre-training setting
    if configs.fine_tune:
        print("Time: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') +
              'Fine tuning based on pre-trained model: {}.'.format(configs.pretrained_model))
        # contune training
        model.load_state_dict(t.load(configs.model_save_dir + '/' + configs.pretrained_model + '.pth'))
    else:
        print("Time: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') +
              'Learning from scratch. Initializing the model params...')
        ini_model_params(model, configs.ini_mode)
    print("Time: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "  Load Model Successfully")

    # GPU setting
    if configs.use_gpu:
        print("Time: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ' Using GPU... ids: \t{}'.format(
            str(configs.device_ids)))
        device = t.device('cuda:0')
        if len(configs.device_ids) > 1:
            model = nn.DataParallel(model, device_ids=configs.device_ids, dim=0)
            model.to(device)
        else:
            # model.to(device).double()
            model.to(device)
    else:
        print("Time: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + 'Using CPU...')

    """print model scale"""
    # from torchsummary import summary
    # summary(model, (in_len, channel_num, img_width, img_height))
    tensor = t.rand(1, in_len, channel_num, img_width, img_height)
    tensor = tensor.float()
    tensor = tensor.cuda()
    output, ground_truth = model_dict.model_forward(model, in_len, out_len, tensor, tensor, configs)

    # from fvcore.nn import FlopCountAnalysis, parameter_count_table
    # tensor = t.rand(1, in_len, channel_num, img_width, img_height)
    # tensor = tensor.float()
    # tensor = tensor.cuda()
    # # model = model
    # # 分析FLOPs
    # flops = FlopCountAnalysis(model, tensor)
    # # flops = FlopCountAnalysis(model, (tensor, tensor))
    # print("FLOPs: ", flops.total())
    # print(parameter_count_table(model))
    """Load dataloader"""
    train_dataloader, valid_dataloader, test_dataloader = dataset.load_data(configs)
    print("Time: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') +
          'Load dataset successfully...')

    # Definition of loss function
    criterion1 = nn.MSELoss()
    lam1 = 1.0
    criterion2 = nn.L1Loss()
    lam2 = 1.0
    # criterion = lam1 * criterion1 + lam2 * criterion2

    # freeze the pe-trained params of the large model, only tune the visual prompt
    if configs.model == 'MS2Pv3_tune':
        for name, param in model.named_parameters():
            if 'prompt' in name:
                print('visual prompt {} layer is freezed here.'.format(name))
        optimizer = t.optim.Adam([{'params': [param for name, param in model.named_parameters() if 'prompt' in name]}],
                                 lr=configs.learning_rate, betas=configs.optim_betas)
        scheduler = t.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=configs.scheduler_gamma)
    else:
        optimizer = t.optim.Adam(model.parameters(), lr=configs.learning_rate, betas=configs.optim_betas)
        scheduler = t.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=configs.scheduler_gamma)

    # # check the frozen state of mdoel
    # for name, param in model.named_parameters():
    #     print('layer: {} is {}'.format(name, param.requires_grad))

    writer = SummaryWriter(log_dir=configs.log_dir)
    valid_log_path = os.path.join(configs.log_dir, 'valid_record.txt')
    valid_log = open(valid_log_path, 'w')

    print("Time: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "  Start Train")
    print("Time: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "  Start Train", file=valid_log)
    train_global_step = 0
    max_score = 0

    """Training & Validation in epochs"""
    for epoch in range(configs.train_max_epoch):
        # epoch += 32
        """Training"""
        for iter, data in enumerate(train_dataloader):
            # randomly sampling
            if configs.random_sampling:
                if iter * configs.batch_size >= configs.random_iters:
                    break

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
            optimizer.zero_grad()

            # Prepare output (generating output on input) and ground-truth
            if configs.model in ['ConvLSTMNet', 'PredRNN', 'PredRNN_patch', 'UNet', 'ConvLSTM', 'MIM', 'Eidetic3DLSTM',
                                 'MotionRNN', 'TrajGRU', 'PredRNNpp', 'LMC', 'SimVP', 'SimVPv2', 'MS2P', 'MS2Pv2',
                                 'MS2Pv3', 'MS2Pv3_tune']:
                output, ground_truth = model_dict.model_forward(model, in_len, out_len, input, ground_truth, configs)
            elif configs.model == 'PhyDNet':
                constraints = t.zeros((49, 7, 7)).to(device)
                ind = 0
                for i in range(0, 7):
                    for j in range(0, 7):
                        constraints[ind, i, j] = 1
                        ind += 1
                loss = 0
                for ei in range(in_len - 1):
                    encoder_output, encoder_hidden, output_image, _, _ = model(input[:, ei, :, :, :],
                                                                               (ei == 0))
                    loss += criterion1(output_image, ground_truth[:, ei, :, :, :]) + criterion2(output_image,
                                                                                                ground_truth[:, ei, :,
                                                                                                :, :])

                decoder_input = input[:, -1, :, :, :]  # first decoder input = last image of input sequence
                for di in range(out_len):
                    decoder_output, decoder_hidden, output_image, _, _ = model(decoder_input)
                    target = ground_truth[:, di, :, :, :]
                    loss += lam1 * criterion1(output_image, target) + lam2 * criterion2(output_image, target)
                    decoder_input = output_image

                k2m = K2M([7, 7]).to(device)
                for b in range(0, model.phycell.cell_list[0].input_dim):
                    filters = model.phycell.cell_list[0].F.conv1.weight[:, b, :, :]  # (nb_filters,7,7)
                    m = k2m(filters.double())
                    m = m.float()
                    loss += lam1 * criterion1(m, constraints) + lam2 * criterion2(m, constraints)

            # for self-supervised pre-training
            # ground_truth = input
            # Calculate loss and backward
            # loss1 = weighted_l2_loss_radar(output, ground_truth)
            # loss2 = vgg_loss(output, ground_truth)
            if configs.model == 'PhyDNet':
                loss.backward()
                optimizer.step()
                train_global_step += 1
            elif configs.model == 'MS2Pv2' or configs.model == 'MS2Pv3':
                loss1 = criterion1(output[0], ground_truth) + criterion2(output[0], ground_truth)
                loss2 = criterion1(output[1], input) + criterion2(output[1], input)
                loss = lam1 * loss1 + lam2 * loss2
                loss.backward()
                optimizer.step()
                train_global_step += 1
            elif configs.model == 'MS2Pv3_tune':
                loss1 = criterion1(output, ground_truth)
                loss2 = criterion2(output, ground_truth)
                # for pre-training
                loss = lam1 * loss1 + lam2 * loss2
                loss.backward()
                optimizer.step()
                train_global_step += 1
            else:
                loss1 = criterion1(output, ground_truth)
                loss2 = criterion2(output, ground_truth)
                # for pre-training
                loss = lam1 * loss1 + lam2 * loss2
                loss.backward()
                optimizer.step()
                train_global_step += 1

            # Print loss & log loss
            if (iter + 1) % configs.train_print_fre == 0:
                print('Time: ' + datetime.now().strftime(
                    '%Y-%m-%d %H:%M:%S') + '  Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format((epoch + 1), (
                        iter + 1) * configs.batch_size, len(train_dataloader) * configs.batch_size, 100. * (iter + 1) / len(
                    train_dataloader), loss.item()))
                writer.add_scalar('Train/Loss/loss', loss.item(), train_global_step)
            # break

            # save by iters
            # if not os.path.exists(configs.model_save_dir):
            #     os.makedirs(configs.model_save_dir)
            #
            # if (iter * configs.batch_size) % configs.model_save_iter == 0:
            #     if len(configs.device_ids) > 1:
            #         t.save(model.module.state_dict(),
            #                configs.model_save_dir + '/' + configs.model + '_iter' + str(
            #                    iter * configs.batch_size) + '.pth')
            #     else:
            #         print('saving')
            #         t.save(model.state_dict(),
            #                configs.model_save_dir + '/' + configs.model + '_iter' + str(
            #                    iter * configs.batch_size) + '.pth')

        # Save by epochs
        if not os.path.exists(configs.model_save_dir):
            os.makedirs(configs.model_save_dir)

        if (epoch + 1) % configs.model_save_fre == 0:
            if len(configs.device_ids) > 1:
                t.save(model.module.state_dict(),
                       configs.model_save_dir + '/' + configs.model + '_epoch' + str(epoch + 1) + '.pth')
            else:
                t.save(model.state_dict(),
                       configs.model_save_dir + '/' + configs.model + '_epoch' + str(epoch + 1) + '.pth')

        model_name = configs.model_save_dir + '/' + configs.model + '_epoch' + str(epoch + 1) + '.pth'

        """Validate"""
        # if configs.dataset_type in ['hko7', 'shanghai2020', 'sevir-vil', 'sevir-vis']:
        #     pod, far, csi, bias, hss, ssim = test(test_dataloader, mode='weather', model_name=model_name, dBZ_threshold=10)
        # else:
        #     mse, psnr, ssim, lpips = test(test_dataloader, mode='simple', model_name=model_name)

    writer.close()


def main():
    # model_list = ['UNet', 'PredRNN', 'PredRNNpp', 'ConvLSTM', 'ConvLSTMNet', 'MotionRNN', 'SimVP']
    # model_list = ['UNet', 'PredRNN', 'PredRNNpp', 'ConvLSTM', 'ConvLSTMNet', 'MotionRNN', 'SimVP']
    # negative valid
    # for ele in model_list:
    #     configs.model = ele

    if configs.mode == 'train':
        train()
    elif configs.mode == 'test':
        test(mode=configs.eval_mode)


if __name__ == '__main__':
    main()
