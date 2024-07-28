import torch as t


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
    score = t.where(t.isnan(score), t.full_like(score, 0.0), score)
    return score, index.int()


def forecast_skill_score_precip(output, ground_truth):
    '''

    :param output: (batch_size, 4, 480, 560)
    :param ground_truth: (batch_size, 4, 480, 560)
    :return:
    '''
    out1 = (output[:, 0, :, :] + output[:, 1, :, :]) / 2
    truth1 = (ground_truth[:, 0, :, :] + ground_truth[:, 1, :, :]) / 2
    out2 = (output[:, 1, :, :] + output[:, 2, :, :]) / 2
    truth2 = (ground_truth[:, 1, :, :] + ground_truth[:, 2, :, :]) / 2
    out3 = (output[:, 2, :, :] + output[:, 3, :, :]) / 2
    truth3 = (ground_truth[:, 2, :, :] + ground_truth[:, 3, :, :]) / 2

    out1_score1, out1_index1 = cal_skill_score(out1, truth1, 5 / 100, 10 / 100)
    out1_score2, out1_index2 = cal_skill_score(out1, truth1, 10 / 100, 20 / 100)
    out1_score3, out1_index3 = cal_skill_score(out1, truth1, 20 / 100, 1.0, lt=False)

    out2_score1, out2_index1 = cal_skill_score(out2, truth2, 5 / 100, 10 / 100)
    out2_score2, out2_index2 = cal_skill_score(out2, truth2, 10 / 100, 20 / 100)
    out2_score3, out2_index3 = cal_skill_score(out2, truth2, 20 / 100, 1.0, lt=False)

    out3_score1, out3_index1 = cal_skill_score(out3, truth3, 5 / 100, 10 / 100)
    out3_score2, out3_index2 = cal_skill_score(out3, truth3, 10 / 100, 20 / 100)
    out3_score3, out3_index3 = cal_skill_score(out3, truth3, 20 / 100, 1.0, lt=False)

    return out1_score1, out1_index1, out1_score2, out1_index2, out1_score3, out1_index3, out2_score1, out2_index1, out2_score2, out2_index2, out2_score3, out2_index3, out3_score1, out3_index1, out3_score2, out3_index2, out3_score3, out3_index3
