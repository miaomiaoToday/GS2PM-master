import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
from imageio import imwrite
from skimage.transform import resize
from PIL import Image
# import numpy as np

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import xlwt
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
# from configs import configs
from einops import rearrange
import math

import matplotlib
matplotlib.use("Agg")

import torch

def img_seq_summary(img_seq, global_step, name_scope, writer):
    seq_len = img_seq.size()[0]
    for i in range(seq_len):
        writer.add_images(name_scope + '/Img' + str(i + 1), img_seq[i], global_step)


def save_test_results(log_dir, pod, far, csi, bias, hss, ssim=None):
    test_results_path = os.path.join(log_dir, 'test_results.xls')
    work_book = xlwt.Workbook(encoding='utf-8')
    sheet = work_book.add_sheet('sheet')
    sheet.write(0, 0, 'pod')
    for col, label in enumerate(pod.tolist()):
        sheet.write(0, 1 + col, str(label))
    sheet.write(1, 0, 'far')
    for col, label in enumerate(far.tolist()):
        sheet.write(1, 1 + col, str(label))
    sheet.write(2, 0, 'csi')
    for col, label in enumerate(csi.tolist()):
        sheet.write(2, 1 + col, str(label))
    sheet.write(3, 0, 'bias')
    for col, label in enumerate(bias.tolist()):
        sheet.write(3, 1 + col, str(label))
    sheet.write(4, 0, 'hss')
    for col, label in enumerate(hss.tolist()):
        sheet.write(4, 1 + col, str(label))
    if ssim is not None:
        sheet.write(5, 0, 'ssim')
        for col, label in enumerate(ssim.tolist()):
            sheet.write(5, 1 + col, str(label))
    work_book.save(test_results_path)


def save_test_imgs(log_dir, index, input, ground_truth, output, dataset='HKO_7'):
    if dataset == 'HKO':
        input = 70.0 * input - 10.0
        output = 70.0 * output - 10.0
        ground_truth = 70.0 * ground_truth - 10.0

    input_seq_len = input.size()[1]
    out_seq_len = output.size()[1]
    height = input.size()[4]
    width = input.size()[3]
    x = np.arange(0, width)
    y = np.arange(height, 0, -1)
    levels = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65]
    cmp = mpl.colors.ListedColormap(['white', 'lightskyblue', 'cyan', 'lightgreen', 'limegreen', 'green',
                                     'yellow', 'orange', 'chocolate', 'red', 'firebrick', 'darkred', 'fuchsia',
                                     'purple'], 'indexed')
    if not os.path.exists(os.path.join(log_dir, 'sample' + str(index + 1))):
        os.makedirs(os.path.join(log_dir, 'sample' + str(index + 1)))
    for i in range(input_seq_len):
        img = input[:, i].squeeze().cpu().numpy()

        plt.contourf(x, y, img, levels=levels, extend='both', cmap=cmp)
        save_fig_path = os.path.join(log_dir, 'sample' + str(index + 1), 'input' + str(i + 1))
        plt.xticks([])
        plt.yticks([])
        plt.savefig(save_fig_path, bbox_inches='tight', dpi=600)
        plt.clf()
    for i in range(out_seq_len):
        img = output[:, i].squeeze().cpu().numpy()  # use [:, i] to replace [i], for the special seq: [bs, l, c, w, h]
        plt.contourf(x, y, img, levels=levels, extend='both', cmap=cmp)
        save_fig_path = os.path.join(log_dir, 'sample' + str(index + 1), 'output' + str(i + 1))
        plt.xticks([])
        plt.yticks([])
        plt.savefig(save_fig_path, bbox_inches='tight', dpi=600)
        plt.clf()
    for i in range(out_seq_len):
        img = ground_truth[:, i].squeeze().cpu().numpy()   # the same as above
        plt.contourf(x, y, img, levels=levels, extend='both', cmap=cmp)
        save_fig_path = os.path.join(log_dir, 'sample' + str(index + 1), 'ground_truth' + str(i + 1))
        plt.xticks([])
        plt.yticks([])
        plt.savefig(save_fig_path, bbox_inches='tight', dpi=600)
        plt.clf()
    return


def get_cmap(type,encoded=True):
    if type.lower()=='vis':
        cmap,norm = vis_cmap(encoded)
        vmin,vmax=(0,10000) if encoded else (0,1)
    elif type.lower()=='vil':
        cmap,norm=vil_cmap(encoded)
        vmin,vmax=None,None
    elif type.lower()=='precip':
        cmap, norm = precip_cmap()
        vmin, vmax = None, None

    return cmap,norm,vmin,vmax


def trans2kg(X):
    num = None
    if X <= 5:
        num = 0
    if 5 < X <= 18:
        num = (X - 2) / 90.66
    if 18 < X <= 255:
        num = math.exp((X - 83.9) / 38.9)
    return num


def pixel_to_dBZ(img):
    """

    Parameters
    ----------
    img : np.ndarray or float

    Returns
    -------

    """
    return img * 70.0 - 10.0


def pixel_to_rainfall(img, a=None, b=None):
    """Convert the pixel values to real rainfall intensity

    Parameters
    ----------
    img : np.ndarray
    a : float32, optional
    b : float32, optional

    Returns
    -------
    rainfall_intensity : np.ndarray
    """
    # img = img.cpu().numpy()
    if a is None:
        a = 58.53
    if b is None:
        b = 1.56
    dBZ = pixel_to_dBZ(img)
    dBR = (dBZ - 10.0 * np.log10(a)) / b
    rainfall_intensity = torch.pow(10, dBR / 10.0)
    return rainfall_intensity


def precip_cmap(encoded=True):
    cols = ['#CCFF99', '#CCFF99', '#a6f28e', '#3dba3d', '#61b8ff', '#8000e1', '#fa00fa', '#800040', '#800040']
    lev = [0, 0.1, 0.5, 1, 2, 5, 10, 20, 50]
    #TODO:  encoded=False
    nil = cols.pop(0)
    under = cols[0]
    over = cols.pop()
    cmap=mpl.colors.ListedColormap(cols)
    cmap.set_bad(nil)
    cmap.set_under(under)
    cmap.set_over(over)
    norm = mpl.colors.BoundaryNorm(lev, cmap.N)
    return cmap,norm


def vil_cmap(encoded=True):
    cols=[   [0,0,0],
             [0.30196078431372547, 0.30196078431372547, 0.30196078431372547],
             [0.1568627450980392,  0.7450980392156863,  0.1568627450980392],
             [0.09803921568627451, 0.5882352941176471,  0.09803921568627451],
             [0.0392156862745098,  0.4117647058823529,  0.0392156862745098],
             [0.0392156862745098,  0.29411764705882354, 0.0392156862745098],
             [0.9607843137254902,  0.9607843137254902,  0.0],
             [0.9294117647058824,  0.6745098039215687,  0.0],
             [0.9411764705882353,  0.43137254901960786, 0.0],
             [0.6274509803921569,  0.0, 0.0],
             [0.9058823529411765,  0.0, 1.0]]
    lev = [16.0, 31.0, 59.0, 74.0, 100.0, 133.0, 160.0, 181.0, 219.0, 255]
    #TODO:  encoded=False
    nil = cols.pop(0)
    under = cols[0]
    over = cols.pop()
    cmap=mpl.colors.ListedColormap(cols)
    cmap.set_bad(nil)
    cmap.set_under(under)
    cmap.set_over(over)
    norm = mpl.colors.BoundaryNorm(lev, cmap.N)
    return cmap,norm


def vis_cmap(encoded=True):
    cols=[[0,0,0],
             [0.0392156862745098, 0.0392156862745098, 0.0392156862745098],
             [0.0784313725490196, 0.0784313725490196, 0.0784313725490196],
             [0.11764705882352941, 0.11764705882352941, 0.11764705882352941],
             [0.1568627450980392, 0.1568627450980392, 0.1568627450980392],
             [0.19607843137254902, 0.19607843137254902, 0.19607843137254902],
             [0.23529411764705882, 0.23529411764705882, 0.23529411764705882],
             [0.27450980392156865, 0.27450980392156865, 0.27450980392156865],
             [0.3137254901960784, 0.3137254901960784, 0.3137254901960784],
             [0.35294117647058826, 0.35294117647058826, 0.35294117647058826],
             [0.39215686274509803, 0.39215686274509803, 0.39215686274509803],
             [0.43137254901960786, 0.43137254901960786, 0.43137254901960786],
             [0.47058823529411764, 0.47058823529411764, 0.47058823529411764],
             [0.5098039215686274, 0.5098039215686274, 0.5098039215686274],
             [0.5490196078431373, 0.5490196078431373, 0.5490196078431373],
             [0.5882352941176471, 0.5882352941176471, 0.5882352941176471],
             [0.6274509803921569, 0.6274509803921569, 0.6274509803921569],
             [0.6666666666666666, 0.6666666666666666, 0.6666666666666666],
             [0.7058823529411765, 0.7058823529411765, 0.7058823529411765],
             [0.7450980392156863, 0.7450980392156863, 0.7450980392156863],
             [0.7843137254901961, 0.7843137254901961, 0.7843137254901961],
             [0.8235294117647058, 0.8235294117647058, 0.8235294117647058],
             [0.8627450980392157, 0.8627450980392157, 0.8627450980392157],
             [0.9019607843137255, 0.9019607843137255, 0.9019607843137255],
             [0.9411764705882353, 0.9411764705882353, 0.9411764705882353],
             [0.9803921568627451, 0.9803921568627451, 0.9803921568627451],
             [0.9803921568627451, 0.9803921568627451, 0.9803921568627451]]
    lev=np.array([0.  , 0.02, 0.04, 0.06, 0.08, 0.1 , 0.12, 0.14, 0.16, 0.2 , 0.24,
       0.28, 0.32, 0.36, 0.4 , 0.44, 0.48, 0.52, 0.56, 0.6 , 0.64, 0.68,
       0.72, 0.76, 0.8 , 0.9 , 1.  ])
    if encoded:
        lev*=1e4
    nil = cols[0]
    under = cols[0]
    over = cols.pop()
    cmap=mpl.colors.ListedColormap(cols)
    cmap.set_bad(nil)
    cmap.set_under(under)
    cmap.set_over(over)
    norm = mpl.colors.BoundaryNorm(lev, cmap.N)
    return cmap,norm


def save_travel_imgs(log_dir, index, input, ground_truth, dataset='hko7', save_mode='simple'):
    # if save_mode is simple, original imgs, else if is color, colored for weather.
    if save_mode == 'simple':
        if dataset == 'hko7':
            input = 255 * input
            ground_truth = 255 * ground_truth
        if dataset == 'shanghai2020':
            input = 255 * input
            ground_truth = 255 * ground_truth
        if dataset == 'bair':
            input = 255 * input
            ground_truth = 255 * ground_truth
        if dataset == 'human':
            input = 255 * input
            ground_truth = 255 * ground_truth
        if dataset == 'moving_mnist':
            input = 255 * input
            ground_truth = 255 * ground_truth
        if dataset == 'taxibj':
            input = 255 * input
            ground_truth = 255 * ground_truth
        if dataset == 'kitticaltech':
            input = 255 * input
            ground_truth = 255 * ground_truth
        if dataset == 'kth':
            input = 255 * input
            ground_truth = 255 * ground_truth
        if dataset == 'sevir-vil':
            input = 255 * input
            ground_truth = 255 * ground_truth
        if dataset == 'sevir-vis':      # this is broken because the values are 1e4 bits and 256 is not enough.
            input = 10000 * input
            ground_truth = 10000 * ground_truth
        if dataset == 'pam':
            input = 255 * input
            ground_truth = 255 * ground_truth
    elif save_mode == 'weather':
        if dataset == 'hko7':
            input = 255 * input
            ground_truth = 255 * ground_truth
            v_cmap, v_norm, v_vmin, v_vmax = get_cmap('vil', encoded=True)
        if dataset == 'shanghai2020':
            input = 255 * input
            ground_truth = 255 * ground_truth
            v_cmap, v_norm, v_vmin, v_vmax = get_cmap('vil', encoded=True)
        if dataset == 'sevir-vil':
            input = 255 * input
            ground_truth = 255 * ground_truth
            v_cmap, v_norm, v_vmin, v_vmax = get_cmap('vil', encoded=True)
        if dataset == 'sevir-vis':
            input = 10000 * input
            ground_truth = 10000 * ground_truth
            v_cmap, v_norm, v_vmin, v_vmax = get_cmap('vis', encoded=True)
        if dataset == 'pam':
            input = 255 * input
            ground_truth = 255 * ground_truth
            v_cmap, v_norm, v_vmin, v_vmax = get_cmap('vil', encoded=True)

        # fig, axs = plt.subplots(1, 5, figsize=(14, 5))
        # frame_idx = 30
        # axs[0].imshow(vis[:, :, frame_idx], cmap=vis_cmap, norm=vis_norm, vmin=vis_vmin, vmax=vis_vmax), axs[
        #     0].set_title('VIS')
        # axs[3].imshow(vil[:, :, frame_idx], cmap=vil_cmap, norm=vil_norm, vmin=vil_vmin, vmax=vil_vmax), axs[
        #     3].set_title('VIL')


        # input_seq_len = input.size()[1]
        # out_seq_len = output.size()[1]
        # height = input.size()[4]
        # width = input.size()[3]
        # x = np.arange(0, width)
        # y = np.arange(height, 0, -1)
        # levels = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65]
        # cmp = mpl.colors.ListedColormap(['white', 'lightskyblue', 'cyan', 'lightgreen', 'limegreen', 'green',
        #                                  'yellow', 'orange', 'chocolate', 'red', 'firebrick', 'darkred', 'fuchsia',
        #                                  'purple'], 'indexed')
        # if not os.path.exists(os.path.join(log_dir, 'sample' + str(index + 1))):
        #     os.makedirs(os.path.join(log_dir, 'sample' + str(index + 1)))
        # for i in range(input_seq_len):
        #     # x = rearrange(x, 'b l c h w -> (b l) c h w')   # set batchsize = 1, and adapt the order or [:, i]{batchsize=1, seqlen}
        #     img = input[:, i].squeeze().cpu().numpy()
        #
        #     plt.contourf(x, y, img, levels=levels, extend='both', cmap=cmp)
        #     save_fig_path = os.path.join(log_dir, 'sample' + str(index + 1), 'input' + str(i + 1))
        #     if save_mode == 'simple':
        #         plt.xticks([])
        #         plt.yticks([])
        #         plt.savefig(save_fig_path, bbox_inches='tight', dpi=600)
        #     elif save_mode == 'integral':
        #         plt.title('Input')
        #         plt.xlabel('Timestep' + str(i + 1))
        #         plt.colorbar()
        #         plt.savefig(save_fig_path, dpi=600)
        #     plt.clf()

    input_seq_len = input.size()[1]
    out_seq_len = ground_truth.size()[1]
    height = input.size()[4]
    width = input.size()[3]

    if not os.path.exists(os.path.join(log_dir, 'sample' + str(index + 1))):
        os.makedirs(os.path.join(log_dir, 'sample' + str(index + 1)))
    for i in range(input_seq_len):
        img = input[:, i].squeeze().cpu().numpy()
        if dataset not in ['hko7', 'shanghai2020', 'moving_mnist', 'kth', 'sevir-vil', 'sevir-vis']:
            img = rearrange(img, 'c h w -> h w c')

        # if dataset not in ['hko7', 'shanghai2020']:
        #     img = rearrange(img, 'h w c -> c h w')
        # img = Image.fromarray(img.astype(np.uint8))
        save_fig_path = os.path.join(log_dir, 'sample' + str(index + 1), 'input' + str(i + 1) + '.png')
        if save_mode == 'simple':
            img = img.astype(np.uint8)
            img = Image.fromarray(img)
            img.save(save_fig_path)
        elif save_mode == 'weather':
            fig, axs = plt.subplots(1, 1, figsize=(5, 5))
            plt.xticks([]), plt.yticks([])
            axs.imshow(img, cmap=v_cmap, norm=v_norm)
            # plt.savefig(save_fig_path, dpi=600)
            plt.savefig(save_fig_path, bbox_inches='tight', dpi=300)
        plt.clf()

    for i in range(out_seq_len):
        img = ground_truth[:, i].squeeze().cpu().numpy()   # the same as above
        if dataset not in ['hko7', 'shanghai2020', 'moving_mnist', 'kth', 'sevir-vil', 'sevir-vis']:
            img = rearrange(img, 'c h w -> h w c')
        # img = Image.fromarray(img.astype(np.uint8))
        save_fig_path = os.path.join(log_dir, 'sample' + str(index + 1), 'ground_truth' + str(i + 1) + '.png')
        if save_mode == 'simple':
            img = img.astype(np.uint8)
            img = Image.fromarray(img)
            img.save(save_fig_path)
        elif save_mode == 'weather':
            fig, axs = plt.subplots(1, 1, figsize=(5, 5))
            plt.xticks([]), plt.yticks([])
            axs.imshow(img, cmap=v_cmap, norm=v_norm)
            # plt.savefig(save_fig_path, dpi=600)
            plt.savefig(save_fig_path, bbox_inches='tight', dpi=300)
        plt.clf()
    return


def save_test_imgs(log_dir, index, input, output, ground_truth, dataset='hko7', save_mode='simple'):
    # if save_mode is simple, original imgs, else if is color, colored for weather.
    if save_mode == 'simple':
        if dataset == 'hko7':
            input = 255 * input
            output = 255 * output
            ground_truth = 255 * ground_truth
        if dataset == 'shanghai2020':
            input = 255 * input
            output = 255 * output
            ground_truth = 255 * ground_truth
        if dataset == 'bair':
            input = 255 * input
            output = 255 * output
            ground_truth = 255 * ground_truth
        if dataset == 'human':
            input = 255 * input
            output = 255 * output
            ground_truth = 255 * ground_truth
        if dataset == 'moving_mnist':
            input = 255 * input
            output = 255 * output
            ground_truth = 255 * ground_truth
        if dataset == 'taxibj':
            input = 255 * input
            output = 255 * output
            ground_truth = 255 * ground_truth
        if dataset == 'kitticaltech':
            input = 255 * input
            output = 255 * output
            ground_truth = 255 * ground_truth
        if dataset == 'kitticaltech_long':
            input = 255 * input
            output = 255 * output
            ground_truth = 255 * ground_truth
        if dataset == 'kth':
            input = 255 * input
            output = 255 * output
            ground_truth = 255 * ground_truth
        if dataset == 'sevir-vil':
            input = 255 * input
            output = 255 * output
            ground_truth = 255 * ground_truth
        if dataset == 'sevir-vis':      # this is broken because the values are 1e4 bits and 256 is not enough.
            input = 10000 * input
            output = 10000 * output
            ground_truth = 10000 * ground_truth
        if dataset == 'pam':
            input = 255 * input
            output = 255 * output
            ground_truth = 255 * ground_truth
    elif save_mode == 'weather':
        if dataset == 'hko7':
            input = 255 * input
            output = 255 * output
            ground_truth = 255 * ground_truth
            v_cmap, v_norm, v_vmin, v_vmax = get_cmap('vil', encoded=True)
        if dataset == 'shanghai2020':
            input = 255 * input
            output = 255 * output
            ground_truth = 255 * ground_truth
            v_cmap, v_norm, v_vmin, v_vmax = get_cmap('vil', encoded=True)
        if dataset == 'sevir-vil':
            input = 255 * input
            output = 255 * output
            ground_truth = 255 * ground_truth
            v_cmap, v_norm, v_vmin, v_vmax = get_cmap('vil', encoded=True)
        if dataset == 'sevir-vis':
            input = 10000 * input
            output = 10000 * output
            ground_truth = 10000 * ground_truth
            v_cmap, v_norm, v_vmin, v_vmax = get_cmap('vis', encoded=True)
        if dataset == 'pam':
            input = 255 * input
            output = 255 * output
            ground_truth = 255 * ground_truth
            v_cmap, v_norm, v_vmin, v_vmax = get_cmap('vil', encoded=True)
    elif save_mode == 'precip':
        input = pixel_to_rainfall(input)
        output = pixel_to_rainfall(output)
        ground_truth = pixel_to_rainfall(ground_truth)
        v_cmap, v_norm, v_vmin, v_vmax = get_cmap('precip')

    input_seq_len = input.size()[1]
    out_seq_len = ground_truth.size()[1]
    height = input.size()[4]
    width = input.size()[3]

    if not os.path.exists(os.path.join(log_dir, 'sample' + str(index + 1))):
        os.makedirs(os.path.join(log_dir, 'sample' + str(index + 1)))
    for i in range(input_seq_len):
        img = input[:, i].squeeze().cpu().numpy()
        if dataset not in ['hko7', 'shanghai2020', 'moving_mnist', 'kth', 'sevir-vil', 'sevir-vis']:
            img = rearrange(img, 'c h w -> h w c')

        # if dataset == 'pam':
        #     img = rearrange(img, 'c h w -> h w c')

        # if dataset not in ['hko7', 'shanghai2020']:
        #     img = rearrange(img, 'h w c -> c h w')
        # img = Image.fromarray(img.astype(np.uint8))
        save_fig_path = os.path.join(log_dir, 'sample' + str(index + 1), 'input' + str(i + 1) + '.png')

        if save_mode == 'simple':
            img = img.astype(np.uint8)
            img = Image.fromarray(img)
            img.save(save_fig_path)
        elif save_mode == 'weather':
            fig, axs = plt.subplots(1, 1, figsize=(5, 5))
            plt.xticks([]), plt.yticks([])
            # axs.imshow(img[:, :, 0], cmap=v_cmap, norm=v_norm)
            axs.imshow(img[:, :], cmap=v_cmap, norm=v_norm)

            # plt.savefig(save_fig_path, dpi=600)
            plt.savefig(save_fig_path, bbox_inches='tight', dpi=300)
        elif save_mode == 'precip':
            fig, axs = plt.subplots(1, 1, figsize=(5, 5))
            plt.xticks([]), plt.yticks([])
            # axs.imshow(img[:, :, 0], cmap=v_cmap, norm=v_norm)
            axs.imshow(img[:, :], cmap=v_cmap, norm=v_norm)

            # plt.savefig(save_fig_path, dpi=600)
            plt.savefig(save_fig_path, bbox_inches='tight', dpi=300)
        plt.clf()
    for i in range(out_seq_len):
        img = output[:, i].squeeze().cpu().numpy()   # the same as above
        if dataset not in ['hko7', 'shanghai2020', 'moving_mnist', 'kth', 'sevir-vil', 'sevir-vis']:
            img = rearrange(img, 'c h w -> h w c')
        # img = Image.fromarray(img.astype(np.uint8))
        save_fig_path = os.path.join(log_dir, 'sample' + str(index + 1), 'output' + str(i + 1) + '.png')
        if save_mode == 'simple':
            img = img.astype(np.uint8)
            img = Image.fromarray(img)
            img.save(save_fig_path)
        elif save_mode == 'weather':
            fig, axs = plt.subplots(1, 1, figsize=(5, 5))
            plt.xticks([]), plt.yticks([])
            # axs.imshow(img[:, :, 0], cmap=v_cmap, norm=v_norm)
            axs.imshow(img[:, :], cmap=v_cmap, norm=v_norm)
            # plt.savefig(save_fig_path, dpi=600)
            plt.savefig(save_fig_path, bbox_inches='tight', dpi=300)
        elif save_mode == 'precip':
            fig, axs = plt.subplots(1, 1, figsize=(5, 5))
            plt.xticks([]), plt.yticks([])
            # axs.imshow(img[:, :, 0], cmap=v_cmap, norm=v_norm)
            axs.imshow(img[:, :], cmap=v_cmap, norm=v_norm)

            # plt.savefig(save_fig_path, dpi=600)
            plt.savefig(save_fig_path, bbox_inches='tight', dpi=300)
        plt.clf()
    for i in range(out_seq_len):
        img = ground_truth[:, i].squeeze().cpu().numpy()   # the same as above
        if dataset not in ['hko7', 'shanghai2020', 'moving_mnist', 'kth', 'sevir-vil', 'sevir-vis']:
            img = rearrange(img, 'c h w -> h w c')
        # img = Image.fromarray(img.astype(np.uint8))
        save_fig_path = os.path.join(log_dir, 'sample' + str(index + 1), 'ground_truth' + str(i + 1) + '.png')
        if save_mode == 'simple':
            img = img.astype(np.uint8)
            img = Image.fromarray(img)
            img.save(save_fig_path)
        elif save_mode == 'weather':
            fig, axs = plt.subplots(1, 1, figsize=(5, 5))
            plt.xticks([]), plt.yticks([])
            # axs.imshow(img[:, :, 0], cmap=v_cmap, norm=v_norm)
            axs.imshow(img[:, :], cmap=v_cmap, norm=v_norm)
            # plt.savefig(save_fig_path, dpi=600)
            plt.savefig(save_fig_path, bbox_inches='tight', dpi=300)
        elif save_mode == 'precip':
            fig, axs = plt.subplots(1, 1, figsize=(5, 5))
            plt.xticks([]), plt.yticks([])
            # axs.imshow(img[:, :, 0], cmap=v_cmap, norm=v_norm)
            axs.imshow(img[:, :], cmap=v_cmap, norm=v_norm)

            # plt.savefig(save_fig_path, dpi=600)
            plt.savefig(save_fig_path, bbox_inches='tight', dpi=300)
        plt.clf()
    return
