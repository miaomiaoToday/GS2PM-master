import torch as t
from skimage.metrics import structural_similarity


def cal_skill_scores(output, ground_truth, down_threshold, up_threshold, lt=True):
    output_ = (t.ge(output, down_threshold) & t.lt(output, up_threshold)).int() if lt else (
                t.ge(output, down_threshold) & t.le(output, up_threshold)).int()
    ground_truth_ = (t.ge(ground_truth, down_threshold) & t.lt(ground_truth, up_threshold)).int() if lt else (
            t.ge(ground_truth, down_threshold) & t.le(ground_truth, up_threshold)).int()
    N = output.shape[1] * output.shape[2]
    NA = t.sum(output_ * ground_truth_, dim=[1, 2])
    NC = t.sum(ground_truth_ * (1 - output_), dim=[1, 2])
    NB = t.sum(output_ * (1 - ground_truth_), dim=[1, 2])
    ND = t.sum((1 - output_) * (1 - ground_truth_), dim=[1, 2])
    expect = ((NA + NC) * (NA + NB) + (ND + NC) * (ND + NB)).float() / N
    hss = ((NA + ND) - expect).float() / (N - expect).float()
    bias = (NA + NB).float() / (NA + NC).float()
    score = hss * t.pow(t.exp(-t.abs(1.0 - bias)), 0.2)
    score = t.where(t.isnan(score), t.full_like(score, 0.0), score)
    hss = t.where(t.isnan(hss), t.full_like(hss, 0.0), hss)
    bias = t.where(t.isnan(bias), t.full_like(bias, 0.0), bias)
    bias = t.where(t.isinf(bias), t.full_like(bias, 0.0), bias)
    return hss, bias, score


def forecast_skill_scores_radar(output, ground_truth):
    # out1 = output[:, 13, 0, :, :] * 70
    # truth1 = ground_truth[:, 13, 0, :, :] * 70
    # out2 = (output[:, 15, 0, :, :] + output[:, 16, 0, :, :]) / 2 * 70
    # truth2 = (ground_truth[:, 15, 0, :, :] + ground_truth[:, 16, 0, :, :]) / 2 * 70
    # out3 = output[:, 18, 0, :, :] * 70
    # truth3 = ground_truth[:, 18, 0, :, :] * 70
    out1 = output[:, 28, 0, :, :] * 70
    truth1 = ground_truth[:, 28, 0, :, :] * 70
    out2 = (output[:, 33, 0, :, :] + output[:, 33, 0, :, :]) / 2 * 70
    truth2 = (ground_truth[:, 33, 0, :, :] + ground_truth[:, 33, 0, :, :]) / 2 * 70
    out3 = output[:, 38, 0, :, :] * 70
    truth3 = ground_truth[:, 38, 0, :, :] * 70

    out1_hss1, out1_bias1, out1_score1 = cal_skill_scores(out1, truth1, 30, 40)
    out1_hss2, out1_bias2, out1_score2 = cal_skill_scores(out1, truth1, 40, 50)
    out1_hss3, out1_bias3, out1_score3 = cal_skill_scores(out1, truth1, 50, 70, lt=False)

    out2_hss1, out2_bias1, out2_score1 = cal_skill_scores(out2, truth2, 30, 40)
    out2_hss2, out2_bias2, out2_score2 = cal_skill_scores(out2, truth2, 40, 50)
    out2_hss3, out2_bias3, out2_score3 = cal_skill_scores(out2, truth2, 50, 70, lt=False)

    out3_hss1, out3_bias1, out3_score1 = cal_skill_scores(out3, truth3, 30, 40)
    out3_hss2, out3_bias2, out3_score2 = cal_skill_scores(out3, truth3, 40, 50)
    out3_hss3, out3_bias3, out3_score3 = cal_skill_scores(out3, truth3, 50, 70, lt=False)

    hss = 0.3 * (0.2 * out1_hss1 + 0.3 * out1_hss2 + 0.5 * out1_hss3) + 0.3 * (
                0.2 * out2_hss1 + 0.3 * out2_hss2 + 0.5 * out2_hss3) + 0.4 * (
                      0.2 * out3_hss1 + 0.3 * out3_hss2 + 0.5 * out3_hss3)
    bias = 0.3 * (0.2 * out1_bias1 + 0.3 * out1_bias2 + 0.5 * out1_bias3) + 0.3 * (
                0.2 * out2_bias1 + 0.3 * out2_bias2 + 0.5 * out2_bias3) + 0.4 * (
                      0.2 * out3_bias1 + 0.3 * out3_bias2 + 0.5 * out3_bias3)
    score = 0.3 * (0.2 * out1_score1 + 0.3 * out1_score2 + 0.5 * out1_score3) + 0.3 * (
            0.2 * out2_score1 + 0.3 * out2_score2 + 0.5 * out2_score3) + 0.4 * (
                   0.2 * out3_score1 + 0.3 * out3_score2 + 0.5 * out3_score3)
    return hss, bias, score


def cal_skill_score(output, ground_truth, down_threshold, up_threshold, lt=True):
    output_ = (t.ge(output, down_threshold) & t.lt(output, up_threshold)).int() if lt else (
                t.ge(output, down_threshold) & t.le(output, up_threshold)).int()
    ground_truth_ = (t.ge(ground_truth, down_threshold) & t.lt(ground_truth, up_threshold)).int() if lt else (
            t.ge(ground_truth, down_threshold) & t.le(ground_truth, up_threshold)).int()
    index = t.eq(t.sum(ground_truth_, dim=[1, 2]), 0)
    N = output.shape[1] * output.shape[2]
    NA = t.sum(output_ * ground_truth_, dim=[1, 2])
    NC = t.sum(ground_truth_ * (1 - output_), dim=[1, 2])
    NB = t.sum(output_ * (1 - ground_truth_), dim=[1, 2])
    ND = t.sum((1 - output_) * (1 - ground_truth_), dim=[1, 2])
    expect = ((NA + NC) * (NA + NB) + (ND + NC) * (ND + NB)).float() / N
    hss = ((NA + ND) - expect).float() / (N - expect).float()
    bias = (NA + NB).float() / (NA + NC).float()
    score = hss * t.pow(t.exp(-t.abs(1.0 - bias)), 0.2)
    return score


def forecast_skill_score_radar(output, ground_truth):
    out1 = output[:, 13, 0, :, :] * 70
    truth1 = ground_truth[:, 13, 0, :, :] * 70
    out2 = (output[:, 15, 0, :, :] + output[:, 16, 0, :, :]) / 2 * 70
    truth2 = (ground_truth[:, 15, 0, :, :] + ground_truth[:, 16, 0, :, :]) / 2 * 70
    out3 = output[:, 18, 0, :, :] * 70
    truth3 = ground_truth[:, 18, 0, :, :] * 70

    out1_score1 = cal_skill_scores(out1, truth1, 30, 40)
    out1_score2 = cal_skill_scores(out1, truth1, 40, 50)
    out1_score3 = cal_skill_scores(out1, truth1, 50, 70, lt=False)

    out2_score1 = cal_skill_scores(out2, truth2, 30, 40)
    out2_score2 = cal_skill_scores(out2, truth2, 40, 50)
    out2_score3 = cal_skill_scores(out2, truth2, 50, 70, lt=False)

    out3_score1 = cal_skill_scores(out3, truth3, 30, 40)
    out3_score2 = cal_skill_scores(out3, truth3, 40, 50)
    out3_score3 = cal_skill_scores(out3, truth3, 50, 70, lt=False)

    return out1_score1, out1_score2, out1_score3, out2_score1, out2_score2, out2_score3, out3_score1, out3_score2, out3_score3


def crosstab_evaluate(output, ground_truth, dBZ_downvalue, dBZ_upvalue, dataset='HKO_7'):
    dBZ_output = 70.0 * output
    dBZ_ground_truth = 70.0 * ground_truth
    # if dataset == 'HKO':
    #     dBZ_output = 70.0 * output - 10.0
    #     dBZ_ground_truth = 70.0 * ground_truth - 10.0
    # if dataset == 'shanghai':
    #     dBZ_output = 70.0 * output
    #     dBZ_ground_truth = 70.0 * ground_truth
    if len(output.size()) == 5:  # [seq_len, batch_size, channels=1, height, width]
        dim = [2, 3, 4]
    elif len(output.size()) == 4:  # [seq_len, channels=1, height, width]
        dim = [1, 2, 3]
    elif len(output.size()) == 3:  # [channels=1, height, width]
        dim = [0, 1, 2]
    output_ = (t.ge(dBZ_output, dBZ_downvalue) & t.le(dBZ_output, dBZ_upvalue)).int()
    ground_truth_ = (t.ge(dBZ_ground_truth, dBZ_downvalue) & t.le(dBZ_ground_truth, dBZ_upvalue)).int()
    index = t.eq(t.sum(ground_truth_, dim=dim), 0)  #  find the index where the ground-truth sample has no rainfall preddiction hits

    hits = t.sum(output_ * ground_truth_, dim=dim)
    misses = t.sum(ground_truth_ * (1 - output_), dim=dim)
    false_alarms = t.sum(output_ * (1 - ground_truth_), dim=dim)
    correct_rejections = t.sum((1 - output_) * (1 - ground_truth_), dim=dim)
    pod = hits.float() / (hits + misses).float()
    far = false_alarms.float() / (hits + false_alarms).float()
    csi = hits.float() / (hits + misses + false_alarms).float()
    bias = (hits + false_alarms).float() / (hits + misses).float()
    hss = (2.0 * (hits * correct_rejections - misses * false_alarms)).float() / (
                (hits + misses) * (misses + correct_rejections) + (hits + false_alarms) * (
                    false_alarms + correct_rejections)).float()

    # pod = t.where(t.isnan(pod), t.full_like(pod, 0.0), pod)  # replace the nan in pod to 0  (nan is appeared when hits and misses are both 0)
    far = t.where(t.isnan(far), t.full_like(far, 1.0), far)  # replace the nan in far to 1  (nan is appeared when hits and false alarms are both 0)
    # bias = pod / (1.0 - far)
    # bias = t.where(t.isnan(bias), t.full_like(bias, 0.0), bias)
    # csi = t.where(t.isnan(csi), t.full_like(csi, 0.0), csi)
    # hss = t.where(t.isnan(hss), t.full_like(hss, 0.0), hss)
    pod = t.where(index, t.full_like(pod, 0), pod)
    far = t.where(index, t.full_like(far, 0), far)
    csi = t.where(index, t.full_like(csi, 0), csi)
    hss = t.where(index, t.full_like(hss, 0), hss)
    bias = t.where(index, t.full_like(bias, 0), bias)
    return pod, far, csi, bias, hss, index.int()


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

