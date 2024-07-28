import torch as t
from skimage.metrics import structural_similarity
import cv2
import numpy as np
import torch
import lpips


def PSNR(pred, true, min_max_norm=False):
    """Peak Signal-to-Noise Ratio.

    Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
    """
    mse = np.mean((pred.astype(np.float32) - true.astype(np.float32))**2)
    if mse == 0:
        return float('inf')
    else:
        if min_max_norm:  # [0, 1] normalized by min and max
            return 20. * np.log10(1. / np.sqrt(mse))  # i.e., -10. * np.log10(mse)
        else:
            return 20. * np.log10(255. / np.sqrt(mse))  # [-1, 1] normalized by mean and std


def MSE(pred, true, spatial_norm=False):
    if not spatial_norm:
        return np.mean((pred-true)**2, axis=(0, 1)).sum()
    else:
        norm = pred.shape[-1] * pred.shape[-2] * pred.shape[-3]
        return np.mean((pred-true)**2 / norm, axis=(0, 1)).sum()


class LPIPS(torch.nn.Module):
    """Learned Perceptual Image Patch Similarity, LPIPS.

    Modified from
    https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips_2imgs.py
    """

    def __init__(self, net='alex', use_gpu=True):
        super().__init__()
        assert net in ['alex', 'squeeze', 'vgg']
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.loss_fn = lpips.LPIPS(net=net)
        if use_gpu:
            self.loss_fn.cuda()

    def forward(self, img1, img2):
        # Load images, which are min-max norm to [0, 1]
        img1 = lpips.im2tensor(img1 * 255)  # RGB image from [-1,1]
        img2 = lpips.im2tensor(img2 * 255)
        if self.use_gpu:
            img1, img2 = img1.cuda(), img2.cuda()
        return self.loss_fn.forward(img1, img2).squeeze().detach().cpu().numpy()


def crosstab_evaluate_simple(output, ground_truth):
    output = output.cpu().numpy()
    ground_truth = ground_truth.cpu().numpy()
    dBZ_output = 255 * output
    dBZ_ground_truth = 255 * ground_truth
    pred = dBZ_output
    true = dBZ_ground_truth

    # if len(output.size()) == 5:  # [seq_len, batch_size, channels=1, height, width]
    if len(output.shape) == 5:  # [seq_len, batch_size, channels=1, height, width]
        dim = [2, 3, 4]
    elif len(output.shape) == 4:  # [seq_len, channels=1, height, width]
        dim = [1, 2, 3]
    elif len(output.shape) == 3:  # [channels=1, height, width]
        dim = [0, 1, 2]
    # output_ = (t.ge(dBZ_output, dBZ_downvalue) & t.le(dBZ_output, dBZ_upvalue)).int()
    # ground_truth_ = (t.ge(dBZ_ground_truth, dBZ_downvalue) & t.le(dBZ_ground_truth, dBZ_upvalue)).int()
    # index = t.eq(t.sum(ground_truth_, dim=dim), 0)  #  find the index where the ground-truth sample has no rainfall preddiction hits
    psnr_seq, mse_seq, lpips_seq, ssim_seq = [], [], [], []

    for b in range(pred.shape[0]):
        for f in range(pred.shape[1]):
            # psnr += PSNR(pred[b, f], true[b, f])
            psnr_seq.append(PSNR(pred[b, f], true[b, f]))
    # psnr = psnr / (pred.shape[0] * pred.shape[1])

    for b in range(pred.shape[0]):
        for f in range(pred.shape[1]):
            # mse += MSE(pred[b, f], true[b, f])
            mse_seq.append(MSE(pred[b, f], true[b, f]))
    # mse = mse / (pred.shape[0] * pred.shape[1])

    cal_lpips = LPIPS(net='alex', use_gpu=False)
    pred = pred.transpose(0, 1, 3, 4, 2)
    true = true.transpose(0, 1, 3, 4, 2)
    for b in range(pred.shape[0]):
        for f in range(pred.shape[1]):
            # lpips += cal_lpips(pred[b, f], true[b, f])
            if pred.shape[4] == 2:
                value0 = cal_lpips(pred[b, f, :, :, 0].reshape(pred.shape[2], pred.shape[3], 1), true[b, f, :, :, 0].reshape(pred.shape[2], pred.shape[3], 1))
                value1 = cal_lpips(pred[b, f, :, :, 1].reshape(pred.shape[2], pred.shape[3], 1), true[b, f, :, :, 1].reshape(pred.shape[2], pred.shape[3], 1))
                lpips_seq.append((value0 + value1) / 2)
            else:
                lpips_seq.append(cal_lpips(pred[b, f], true[b, f]))
    # lpips = lpips / (pred.shape[0] * pred.shape[1])

    # lpips_seq = 0

    return mse_seq, psnr_seq, lpips_seq


def compute_ssim(output, ground_truth):
    if len(output.shape) == 5:  # [seq_len, batch_size, channels=1, height, width]
        ssim_seq = []
        seq_len = output.shape[0]
        batch_size = output.shape[1]
        for i in range(seq_len):
            ssim_batch = []
            for j in range(batch_size):
                ssim = structural_similarity(output[i, j, 0], ground_truth[i, j, 0], data_range=1)
                ssim_batch.append(ssim)
            ssim_seq.append(ssim_batch)
    return t.Tensor(ssim_seq)
