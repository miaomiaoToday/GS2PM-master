class Configs(object):
    """1. mode and model name"""
    mode = 'train'         # select from: [train, test, valid(x), travel, create]
    # ConvLSTMNet(4), PredRNN(1), PredRNN_pach(x), UNet(4), ConvLSTM(1), MIM(1),
    # Eidetic3DLSTM(1), MotionRNN(1), TrajGRU(1)(x), PredRNNpp(1), ESCL, LMC(x), SimVP, PhyDNet(on single cuda)
    # sota list: UNet, PredRNN, PredRNNpp, ConvLSTM, MIM(x), ConvLSTMNet, MotionRNN, SimVP, PhyDNet(x), SimVP_aug, SimVPv2, MS2P, MS2Pv2, MS2Pv3, MS2Pv3_tune
    if mode is not 'travel':
        model = 'MS2Pv3_tune'
        ini_mode = 'xavier'
    else:
        model = None
    mid_model = 'tau'
    # mid_model = 'uniformer'
    # gsta [18,598,145], convmixer [], convnext [14,852,353], hornet [18,150,081], mlp [15,361,473], mlpmixer [15,361,473],
    # moga [18,580,521], moganet [18,580,521], poolformer [14,735,617], swin [18,218,753], uniformer [17,314,305],
    # van [17,727,361], vit [18,218,753], tau [17,805,185]

    """2. data name and data root"""
    # select from: [bair, human, kinetics(x), kitticaltech, kth, moving_mnist, taxibj, weather(x), hko7, shanghai2020, sevir-vil, sevir-vis] pam, kitticaltech_long
    dataset_type = 'shanghai2020'
    dataset_root = 'data'       # relativve path from root of project
    # dataset_root = '../data'       # relativve path from root of project
    save_open = True  # save switch for train, val, test, travel
    save_mode = 'simple'   # simple, weather, precip (new added)
    eval_mode = 'simple'    # simple, weather, precip (new added)
    """3. random sampling, optional"""
    random_sampling = False     # default false, random_sampling negative
    random_iters = 300000

    """4. custom input length and output length"""
    custom_len = True          # True for custom setting of in_len and out_len activated, for experimental false, default:True
    in_len = 100
    out_len = 100               # check the official setting of hko7.

    """5. normalized size for exp."""
    custom_shape = False          # All figs resized to normalized size for exp. default:False
    img_width = 128
    img_height = 128

    """6. continue training, or fine tune, the pretrained model should in 'model_save_path'"""
    fine_tune = False
    pretrained_model = 'GS2PM-Large'

    """7. GPU setting, for training and testing"""
    use_gpu = True
    num_workers = 0
    device_ids = [0]
    device_ids_eval = [0]

    """8. hyperparameters"""
    batchsize_dict = {
        'sevir-vis': 8,
        'bair': 8,
        'moving_mnist': 8,
        'shanghai2020': 8,
        'kitticaltech': 8,
        'kitticaltech_long': 8,
        'taxibj': 8,
        'hko7': 8,
        'human': 8,
        'kth': 8,
        'sevir-vil': 8,
        'pam': 8,
    }
    batch_size = batchsize_dict[dataset_type]
    test_batch_size = 1     # for save seqs, please set this one to be 1, and other cases could be like 4...
    travel_batch_size = 1   # must be 1.
    if mode in ['travel', 'create']:
        batch_size = travel_batch_size
        test_batch_size = travel_batch_size
    train_max_epoch = 100
    learning_rate = 1e-4
    optim_betas = (0.9, 0.999)
    # scheduler_gamma = 1.0
    scheduler_gamma = 0.1

    """9. saving paths and save print frequency"""
    train_print_fre = 10
    model_save_fre = 10
    model_save_iter = 100000
    log_dir = r'logdir'
    model_save_dir = r'save_models_formal\pretrained'
    test_imgs_save_dir = r'save_results'
    # travel_imgs_save_dir = r'travel_results'
    travel_imgs_save_dir = r'travel_results_long'
    ourdata_save_dir = r'data\pam\test'
    # domains = ['bair', 'hko7', 'human', 'kitticaltech', 'kth', 'moving_mnist', 'sevir-vil', 'sevir-vis', 'shanghai2020']
    domain = 'shanghai2020'

    import os
    if model == 'MS2Pv3_tune':
        model_save_dir = os.path.join(model_save_dir, domain)


configs = Configs()     #

# torchsummary. UNet [650182656.0 | 486,556], PredRNN [45212237824.0 | 2,708,480], PredRNNpp [62819008512.0 | 3,814,400], ConvLSTM [327312998400 | ], MotionRNN [181696135168 | 11,317,620], SimVP [35079716864 | 15,778,433], SimVPv2 [36435996672 | ]
# 2nd sum. UNet [0.7 | 0.5], PredRNN [45.2 | 0.1], PredRNNpp [62.8 | 0.2], ConvLSTM [327.3 | 20.9], ComLSTMNet [22.1 | 1.4], MotionRNN [ 181.7 | 0.6] SimVP [35.1 | 15.8] SimVPv2 [36.4 | 17.8] # [GFLOPs | M]
# Uniformer [39.3 | 17.3] Ours [69.4 | 38.4] [298.4 | 176.3]
