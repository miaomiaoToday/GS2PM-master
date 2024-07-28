import models
import torch as t


def load_model(configs, in_len, out_len, img_width, img_height, channel_num):
    # Model setting
    if configs.model == 'ConvLSTMNet':
        model = models.ConvLSTMNet(
            input_channels=channel_num,
            output_sigmoid=True,
            # model architecture
            layers_per_block=(3, 3, 3, 3),
            hidden_channels=(32, 48, 48, 32),
            skip_stride=2,
            # convolutional tensor-train layers
            cell=r'convlstm',
            cell_params={
                "order": 3,
                "steps": 3,
                "ranks": 8},
            # convolutional parameters
            kernel_size=3).cuda()
    elif configs.model == 'PredRNN':
        model = models.PredRNN(in_channels=channel_num,
                               hidden_channels_list=[16, 16, 16, 16],
                               kernel_size_list=[3, 3, 3, 3]).to("cuda")
    elif configs.model == 'PredRNN_pach':
        model = models.PredRNN_patch()
    elif configs.model == 'UNet':
        # model = models.UNet()
        model = models.UNet(il=in_len, ol=out_len)
    elif configs.model == 'ConvLSTM':
        model = models.ConvLSTM(in_channels=channel_num, size=(img_width, img_height))
    elif configs.model == 'MIM':
        model = models.MIM(in_channels=channel_num,
                           hidden_channels_list=[4, 4, 4, 4],
                           kernel_size_list=[3, 3, 3, 3])
    elif configs.model == 'Eidetic3DLSTM':
        # model = models.Eidetic3DLSTM(in_channels=1, hidden_channels_list=[4],
        #                              window_length=2, kernel_size=(2, 5, 5))
        #  The sequence length 5:5 is okay for training.
        model = models.Eidetic3DLSTM(in_channels=channel_num, hidden_channels_list=[2],
                                     window_length=2, kernel_size=(2, 3, 3))
    elif configs.model == 'MotionRNN':
        model = models.MotionRNN(in_channels=channel_num,
                                 hidden_channels_list=[32, 32, 32, 32],
                                 kernel_size_list=[3, 3, 3, 3])
    elif configs.model == 'TrajGRU':
        model = models.TrajGRU(in_channels=channel_num)
        # assert x.shape[3] == 128 and x.shape[4] == 128, "当前只支持尺寸为128的张量，具体应用请自行计算反卷积的参数"
    elif configs.model == 'PredRNNpp':
        model = models.PredRNNpp(in_channels=channel_num,
                                 hidden_channels_list=[16, 8, 8],
                                 ghu_hidden_channels=16,
                                 kernel_size_list=[5, 5, 5],
                                 ghu_kernel_size=5)
    elif configs.model == 'ESCL':
        model = models.ESCL(
            input_channels=1,
            output_sigmoid=True,
            # model architecture
            layers_per_block=(3, 3, 3, 3),
            hidden_channels=(32, 48, 48, 32),
            skip_stride=2,
            # convolutional tensor-train layers
            cell=r'convlstm',
            cell_params={
                "order": 3,
                "steps": 3,
                "ranks": 8},
            # convolutional parameters
            kernel_size=3).cuda()
    elif configs.model == 'LMC':
        model = models.Predictor(
            memory_size=100).cuda()
    elif configs.model == 'SimVP':
        # model = models.SimVP(tuple([10, 1, 64, 64]), hid_S=64,
        #                    hid_T=256, N_S=4, N_T=8).cuda()
        model = models.SimVP(tuple([in_len, channel_num, img_width, img_height]), hid_S=64,
                             hid_T=256, N_S=4, N_T=8).cuda()
    elif configs.model == 'PhyDNet':
        device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
        phycell = models.PhyCell(input_shape=(img_width, img_height), input_dim=64, F_hidden_dims=[49], n_layers=1,
                                 kernel_size=(7, 7), device=device)
        convcell = models.PhyConvLSTM(input_shape=(img_width, img_height), input_dim=64, hidden_dims=[128, 128, 64],
                                      n_layers=3,
                                      kernel_size=(3, 3), device=device)
        encoder = models.EncoderRNN(phycell, convcell, device, channel_num)

        # the input size should be changed according to the image size, like 32 & 16
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        model = encoder
        print('phycell ', count_parameters(phycell))
        print('convcell ', count_parameters(convcell))
        print('encoder ', count_parameters(encoder))
    elif configs.model == 'SimVPv2':
        model = models.SimVPv2(tuple([in_len, channel_num, img_width, img_height]), hid_S=64,
                             hid_T=128, N_S=4, N_T=8, model_type=configs.mid_model).cuda()
        # model = models.SimVPv2(tuple([in_len, channel_num, img_width, img_height]), hid_S=64,
        #                        hid_T=1024, N_S=4, N_T=32, model_type=configs.mid_model).cuda()
    elif configs.model == 'MS2P':
        model = models.MS2P(tuple([in_len, channel_num, img_width, img_height]), hid_S=64,
                             hid_T=128, N_S=4, N_T=8, model_type=configs.mid_model).cuda()
    elif configs.model == 'MS2Pv2':
        # model = models.MS2Pv2(tuple([in_len, channel_num, img_width, img_height]), hid_S=64,
        #                      hid_T=128, N_S=4, N_T=8, model_type=configs.mid_model).cuda()
        model = models.MS2Pv2(tuple([in_len, channel_num, img_width, img_height]), hid_S=64,
                              hid_T=512, N_S=4, N_T=64, model_type=configs.mid_model).cuda()
    elif configs.model == 'MS2Pv3':
        # model = models.MS2Pv3(tuple([in_len, channel_num, img_width, img_height]), hid_S=64,
        #                       hid_T=512, N_S=4, N_T=64, model_type=configs.mid_model).cuda()
        # tiny(1)
        model = models.MS2Pv3(tuple([in_len, channel_num, img_width, img_height]), hid_S=64,
                              hid_T=128, N_S=4, N_T=8, model_type=configs.mid_model).cuda()
        # base(3)
        # model = models.MS2Pv3(tuple([in_len, channel_num, img_width, img_height]), hid_S=64,
        #                       hid_T=512, N_S=4, N_T=64, model_type=configs.mid_model).cuda()
        # ss(2)
        # model = models.MS2Pv3(tuple([in_len, channel_num, img_width, img_height]), hid_S=64,
        #                       hid_T=384, N_S=4, N_T=32, model_type=configs.mid_model).cuda()
        # s waiting(4)
        # model = models.MS2Pv3(tuple([in_len, channel_num, img_width, img_height]), hid_S=64,
        #                       hid_T=560, N_S=4, N_T=70, model_type=configs.mid_model).cuda()
        # small
        # model = models.MS2Pv3(tuple([in_len, channel_num, img_width, img_height]), hid_S=64,
        #                       hid_T=768, N_S=4, N_T=64, model_type=configs.mid_model).cuda()
        # small depth
        # model = models.MS2Pv3(tuple([in_len, channel_num, img_width, img_height]), hid_S=64,
        #                       hid_T=512, N_S=4, N_T=96, model_type=configs.mid_model).cuda()
        # small d 2
        # model = models.MS2Pv3(tuple([in_len, channel_num, img_width, img_height]), hid_S=64,
        #                       hid_T=768, N_S=4, N_T=48, model_type=configs.mid_model).cuda()
        # small d 4
        # model = models.MS2Pv3(tuple([in_len, channel_num, img_width, img_height]), hid_S=64,
        #                       hid_T=768, N_S=8, N_T=64, model_type=configs.mid_model).cuda()
        # medium
        # model = models.MS2Pv3(tuple([in_len, channel_num, img_width, img_height]), hid_S=64,
        #                       hid_T=1024, N_S=4, N_T=64, model_type=configs.mid_model).cuda()
        # x1
        # model = models.MS2Pv3(tuple([in_len, channel_num, img_width, img_height]), hid_S=128,
        #                       hid_T=768, N_S=6, N_T=72, model_type=configs.mid_model).cuda()
        # x2
        # model = models.MS2Pv3(tuple([in_len, channel_num, img_width, img_height]), hid_S=80,
        #                       hid_T=640, N_S=4, N_T=72, model_type=configs.mid_model).cuda()
        # 1
        # model = models.MS2Pv3(tuple([in_len, channel_num, img_width, img_height]), hid_S=64,
        #                       hid_T=384, N_S=4, N_T=32, model_type=configs.mid_model).cuda()
    elif configs.model == 'MS2Pv3_tune':
        # tune base
        # model = models.MS2Pv3_tune(tuple([in_len, channel_num, img_width, img_height]), hid_S=64,
        #                       hid_T=512, N_S=4, N_T=64, model_type=configs.mid_model).cuda()
        # model = models.MS2Pv3_tune(tuple([in_len, channel_num, img_width, img_height]), hid_S=64,
        #                            hid_T=560, N_S=4, N_T=70, model_type=configs.mid_model).cuda()
        # tune tiny
        # model = models.MS2Pv3_tune(tuple([in_len, channel_num, img_width, img_height]), hid_S=64,
        #                       hid_T=128, N_S=4, N_T=8, model_type=configs.mid_model).cuda()
        # medium
        model = models.MS2Pv3_tune(tuple([in_len, channel_num, img_width, img_height]), hid_S=64,
                                   hid_T=384, N_S=4, N_T=32, model_type=configs.mid_model).cuda()

    return model


def model_forward(model, in_len, out_len, input, ground_truth, configs):
    # output = None
    if configs.model == 'ConvLSTMNet':
        output = model(inputs=input,
                       input_frames=in_len,
                       future_frames=out_len,
                       output_frames=in_len + out_len - 1,
                       teacher_forcing=False,
                       scheduled_sampling_ratio=1)
        ground_truth = t.cat([input[:, 1:in_len], ground_truth], dim=1)
    elif configs.model == 'PredRNN':
        output = model(input, out_len=out_len)
        ground_truth = ground_truth
    elif configs.model == 'PredRNN_patch':
        output = model(input, in_len=in_len, out_len=out_len)
        ground_truth = t.cat([input[:, 1:in_len], ground_truth], dim=1)
    elif configs.model == 'UNet':
        output = model(input)
        ground_truth = ground_truth
    elif configs.model == 'ConvLSTM':
        output = model(input, out_len=out_len)
        ground_truth = ground_truth
    elif configs.model == 'MIM':
        output = model(input, out_len=out_len)
        ground_truth = ground_truth
    elif configs.model == 'Eidetic3DLSTM':
        output = model(input, out_len=out_len)
        ground_truth = ground_truth
    elif configs.model == 'MotionRNN':
        output = model(input, out_len=out_len)
        ground_truth = ground_truth
    elif configs.model == 'TrajGRU':
        output = model(input, out_len=out_len)
        ground_truth = ground_truth
    elif configs.model == 'PredRNNpp':
        output = model(input, out_len=out_len)
        ground_truth = ground_truth
    # elif configs.model == 'ESCL':
    #     output, output_f1, output_f2 = model(inputs=input,
    #                                          input_frames=in_len,
    #                                          future_frames=out_len,
    #                                          output_frames=in_len + out_len - 1,
    #                                          teacher_forcing=False,
    #                                          scheduled_sampling_ratio=(epoch / configs.train_max_epoch))
    #     ground_truth = t.cat([input[:, 1:in_len], ground_truth], dim=1)
    elif configs.model == 'LMC':
        out_pred = model(input, None, out_len, phase=2)
        output = t.clamp(out_pred, min=0, max=1)
        ground_truth = ground_truth
    elif configs.model == 'SimVP':
        output = model(input)
        # for pre-training
        # output2 = model(input)
        ground_truth = ground_truth
    elif configs.model == 'SimVPv2':
        output = model(input)
        ground_truth = ground_truth
    elif configs.model == 'MS2P':
        output = model(input, ground_truth * 0.000000)
        ground_truth = ground_truth
    elif configs.model == 'MS2Pv2':
        output_pre, output_in = model(input)
        output = [output_pre, output_in]
        ground_truth = ground_truth
    elif configs.model == 'MS2Pv3':
        output_pre, output_in = model(input)
        output = [output_pre, output_in]
        ground_truth = ground_truth
    elif configs.model == 'MS2Pv3_tune':
        output_pre = model(input)
        output = output_pre
        ground_truth = ground_truth

    # if output is None:
    #     print('!!!!!!!!!!!!!!!!')

    return output, ground_truth
