import torch as t
from torch import nn
from einops import rearrange
import random
# import torchvision.models as models
# vgg = models.vgg11(pretrained=True)
# device = t.device('cuda:0')
# vgg.to(device)
# vgg.cuda()
# from torch.nn.functional import adaptive_avg_pool2d
# import torchvision.models as models
# vgg = models.vgg11(pretrained=True)
# vgg.cuda()


def weighted_cross_entropy_loss_radar(logits, ground_truth, use_gpu=True):
    logits = t.log(rearrange(t.clamp(logits, 1e-7, 1.0-1e-7), 'b l c h w -> (b l) c h w'))
    ground_truth = rearrange(ground_truth, 'b l c h w -> (b l) c h w')
    truth_level1 = (t.ge(ground_truth, 0) & t.lt(ground_truth, 30 / 70)).int()
    truth_level2 = (t.ge(ground_truth, 30 / 70) & t.lt(ground_truth, 40 / 70)).int()
    truth_level3 = (t.ge(ground_truth, 40 / 70) & t.lt(ground_truth, 50 / 70)).int()
    truth_level4 = (t.ge(ground_truth, 50 / 70) & t.le(ground_truth, 1)).int()
    label = t.topk(t.cat([truth_level1, truth_level2, truth_level3, truth_level4], dim=1), 1, dim=1)[1].squeeze(1)
    weight = t.tensor([1, 104.59, 172.12, 264.85])
    if use_gpu:
        weight = weight.cuda()
    NLLLoss = nn.NLLLoss2d(weight=weight)
    loss = NLLLoss(logits, label)
    return loss


def weighted_l2_loss_radar(output, ground_truth):
    # random_factor1, random_factor2, random_factor3 = random.uniform(2.5, 7.5), random.uniform(10.0, 20.0), \
    #                                                  random.uniform(25.0, 75.0)
    radar_level1 = t.ge(ground_truth, 30 / 70) & t.lt(ground_truth, 40 / 70)
    radar_level1_matrix = t.where(radar_level1, t.full_like(output, 5.0), t.full_like(output, 1.0))
    radar_level2 = t.ge(ground_truth, 40 / 70) & t.lt(ground_truth, 50 / 70)
    radar_level2_matrix = t.where(radar_level2, t.full_like(output, 10.0), t.full_like(output, 1.0))
    radar_level3 = t.ge(ground_truth, 50 / 70) & t.le(ground_truth, 1.0)
    radar_level3_matrix = t.where(radar_level3, t.full_like(output, 50.0), t.full_like(output, 1.0))
    loss = t.mean(radar_level1_matrix * radar_level2_matrix * radar_level3_matrix * t.pow(output - ground_truth, 2))
    return loss


# def vgg_loss(output, ground_truth):
#     #  batch size. seq len , 1 , w ,h
#     loss = 0
#     fnum = [28, 33, 38]
#     # for i in range(output.shape[0]):
#     for i in fnum:
#         output_b = output[:, i, :]
#         output_b = output_b.repeat(1, 3, 1, 1)
#         output_b = output_b * 255
#         output_features = vgg.features(output_b)[0]
#
#         # output_features = vgg.avgpool(output_features)
#         # output_features = t.flatten(output_features, 1)
#
#         ground_truth_b = ground_truth[:, i, :]
#         ground_truth_b = ground_truth_b.repeat(1, 3, 1, 1)
#         ground_truth_b = ground_truth_b * 255
#         gt_features = vgg.features(ground_truth_b)[0]
#         # gt_features = vgg.avgpool(gt_features)
#         # gt_features = t.flatten(gt_features, 1)
#
#         loss += t.mean(t.pow(output_features - gt_features, 2))
#
#     return loss

