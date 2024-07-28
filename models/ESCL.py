# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under a NVIDIA Open Source Non-commercial license

import torch
import torch.nn as nn
# from utils.convlstmcell import ConvLSTMCell, ConvTTLSTMCell
from .ConvLSTMCell import ConvLSTMCell, ConvTTLSTMCell
import random


## Convolutional-LSTM network
class ESCL(nn.Module):
    def __init__(self,
                 # input to the model
                 input_channels,
                 # architecture of the model
                 layers_per_block, hidden_channels, skip_stride=None,
                 # parameters of convolutional tensor-train layers
                 cell="convlstm", cell_params={},
                 # parameters of convolutional operation
                 kernel_size=3, bias=True,
                 # output function and output format
                 output_sigmoid=False):
        """
        Initialization of a Conv-LSTM network.

        Arguments:
        ----------
        (Hyper-parameters of input interface)
        input_channels: int
            The number of channels for input video.
            Note: 3 for colored video, 1 for gray video.

        (Hyper-parameters of model architecture)
        layers_per_block: list of ints
            Number of Conv-LSTM layers in each block.
        hidden_channels: list of ints
            Number of output channels.
        Note: The length of hidden_channels (or layers_per_block) is equal to number of blocks.

        skip_stride: int
            The stride (in term of blocks) of the skip connections
            default: None, i.e. no skip connection

        [cell_params: dictionary

            order: int
                The recurrent order of convolutional tensor-train cells.
                default: 3
            steps: int
                The number of previous steps used in the recurrent cells.
                default: 5
            rank: int
                The tensor-train rank of convolutional tensor-train cells.
                default: 16
        ]

        (Parameters of convolutional operations)
        kernel_size: int or (int, int)
            Size of the (squared) convolutional kernel.
            default: 3
        bias: bool
            Whether to add bias in the convolutional operation.
            default: True

        (Parameters of the output function)
        output_sigmoid: bool
            Whether to apply sigmoid function after the output layer.
            default: False
        """
        super(ESCL, self).__init__()

        ## Hyperparameters
        self.layers_per_block = layers_per_block
        self.hidden_channels = hidden_channels

        self.num_blocks = len(layers_per_block)
        assert self.num_blocks == len(hidden_channels), "Invalid number of blocks."

        self.skip_stride = (self.num_blocks + 1) if skip_stride is None else skip_stride

        self.output_sigmoid = output_sigmoid

        ## Module type of convolutional LSTM layers

        if cell == "convlstm":  # standard convolutional LSTM
            Cell = lambda in_channels, out_channels: ConvLSTMCell(
                input_channels=in_channels, hidden_channels=out_channels,
                kernel_size=kernel_size, bias=bias)

        elif cell == "convttlstm":  # convolutional tensor-train LSTM
            print('is using convttlstm')
            Cell = lambda in_channels, out_channels: ConvTTLSTMCell(
                input_channels=in_channels, hidden_channels=out_channels,
                order=cell_params["order"], steps=cell_params["steps"], ranks=cell_params["ranks"],
                kernel_size=kernel_size, bias=bias)
        else:
            raise NotImplementedError

        ## Construction of convolutional tensor-train LSTM network

        # stack the convolutional-LSTM layers with skip connections
        self.layers = nn.ModuleDict()
        for b in range(self.num_blocks):
            for l in range(layers_per_block[b]):
                # number of input channels to the current layer
                if l > 0:
                    channels = hidden_channels[b]
                elif b == 0:  # if l == 0 and b == 0:
                    channels = input_channels
                else:  # if l == 0 and b > 0:
                    channels = hidden_channels[b - 1]
                    if b > self.skip_stride:
                        channels += hidden_channels[b - 1 - self.skip_stride]

                lid = "b{}l{}".format(b, l)  # layer ID
                self.layers[lid] = Cell(channels, hidden_channels[b])

        # number of input channels to the last layer (output layer)
        channels = hidden_channels[-1]
        if self.num_blocks >= self.skip_stride:
            channels += hidden_channels[-1 - self.skip_stride]

        self.layers["output"] = nn.Conv2d(channels, input_channels,
                                          kernel_size=1, padding=0, bias=True)

    def forward(self, inputs, input_frames, future_frames, output_frames,
                teacher_forcing=False, scheduled_sampling_ratio=0, checkpointing=False):
        # print('here the shape of inputs is {}'.format(inputs.shape))
        # if teacher_forcing == True:
        #     print('is using teacher forcing')
        """
        Computation of Convolutional LSTM network.

        Arguments:
        ----------
        inputs: a 5-th order tensor of size [batch_size, input_frames, input_channels, height, width]
            Input tensor (video) to the deep Conv-LSTM network.

        input_frames: int
            The number of input frames to the model.
        future_frames: int
            The number of future frames predicted by the model.
        output_frames: int
            The number of output frames returned by the model.

        teacher_forcing: bool
            Whether the model is trained in teacher_forcing mode.
            Note 1: In test mode, teacher_forcing should be set as False.
            Note 2: If teacher_forcing mode is on,  # of frames in inputs = total_steps
                    If teacher_forcing mode is off, # of frames in inputs = input_frames
        scheduled_sampling_ratio: float between [0, 1]
            The ratio of ground-truth frames used in teacher_forcing mode.
            default: 0 (i.e. no teacher forcing effectively)

        Returns:
        --------
        outputs: a 5-th order tensor of size [batch_size, output_frames, hidden_channels, height, width]
            Output frames of the convolutional-LSTM module.
        """
        # cumulative sampling
        cululative_mask = torch.bernoulli(torch.zeros(inputs.size(0), 1, 1, 1, 1,
                                                              device=inputs.device))

        sigma = round(scheduled_sampling_ratio * round(inputs.size(1) / 2))
        steps = max(2, sigma)
        mask_idx1 = [random.randint(0, input_frames-1) for _ in range(steps)]
        mask_idx2 = [random.randint(0, input_frames - 1) for _ in range(steps)]
        # for abs
        mask_idx1 = [0, 1, 2, 3, 4]
        mask_idx2 = [5, 6, 7, 8, 9]

        # compute the teacher forcing mask
        if teacher_forcing and scheduled_sampling_ratio > 1e-6:
            # generate the teacher_forcing mask (4-th order)
            teacher_forcing_mask = torch.bernoulli(scheduled_sampling_ratio *
                                                   torch.ones(inputs.size(0), future_frames - 1, 1, 1, 1,
                                                              device=inputs.device))
        else:  # if not teacher_forcing or scheduled_sampling_ratio < 1e-6:
            teacher_forcing = False

        # the number of time steps in the computational graph
        total_steps = input_frames + future_frames - 1
        outputs = [None] * total_steps

        for t in range(total_steps):
            # input_: 4-th order tensor of size [batch_size, input_channels, height, width]
            # print('check is on: here is times: {}'.format(t))
            if t < input_frames:
                input_ = inputs[:, t]
                # print(inputs.shape)
            elif not teacher_forcing:
                input_ = outputs[t - 1]
            else:  # if t >= input_frames and teacher_forcing:
                # print('is using teacher forcing')
                mask = teacher_forcing_mask[:, t - input_frames]
                input_ = inputs[:, t] * mask + outputs[t - 1] * (1 - mask)

            if t < input_frames:
                if t in mask_idx1:
                    input_2 = inputs[:, t] * cululative_mask[:, 0]
                else:
                    input_2 = inputs[:, t]
                # print(inputs.shape)
            elif not teacher_forcing:
                input_2 = outputs[t - 1]
            else:  # if t >= input_frames and teacher_forcing:
                # print('is using teacher forcing')
                mask = teacher_forcing_mask[:, t - input_frames]
                input_2 = inputs[:, t] * mask + outputs[t - 1] * (1 - mask)

            if t < input_frames:
                if t in mask_idx2:
                    input_3 = inputs[:, t] * cululative_mask[:, 0]
                else:
                    input_3 = inputs[:, t]
                # print(inputs.shape)
            elif not teacher_forcing:
                input_3 = outputs[t - 1]
            else:  # if t >= input_frames and teacher_forcing:
                # print('is using teacher forcing')
                mask = teacher_forcing_mask[:, t - input_frames]
                input_3 = inputs[:, t] * mask + outputs[t - 1] * (1 - mask)

            queue1 = []  # previous outputs for skip connection
            for b in range(self.num_blocks):
                for l in range(self.layers_per_block[b]):
                    lid = "b{}l{}".format(b, l)  # layer ID
                    input_ = self.layers[lid](input_,
                                              first_step=(t == 0), checkpointing=checkpointing)

                queue1.append(input_)
                if b >= self.skip_stride:
                    input_ = torch.cat([input_, queue1.pop(0)], dim=1)  # concat over the channels

            queue2 = []
            # for contrastive learning
            for b in range(self.num_blocks):
                for l in range(self.layers_per_block[b]):
                    lid = "b{}l{}".format(b, l)  # layer ID
                    input_2 = self.layers[lid](input_2,
                                              first_step=(t == 0), checkpointing=checkpointing)

                queue2.append(input_2)
                if b >= self.skip_stride:
                    input_2 = torch.cat([input_2, queue2.pop(0)], dim=1)

            queue3 = []
            for b in range(self.num_blocks):
                for l in range(self.layers_per_block[b]):
                    lid = "b{}l{}".format(b, l)  # layer ID
                    input_3 = self.layers[lid](input_3,
                                              first_step=(t == 0), checkpointing=checkpointing)

                queue3.append(input_3)
                if b >= self.skip_stride:
                    input_3 = torch.cat([input_3, queue3.pop(0)], dim=1)

            # print('1111')
            # map the hidden states to predictive frames (with optional sigmoid function)
            outputs[t] = self.layers["output"](input_)
            if self.output_sigmoid:
                outputs[t] = torch.sigmoid(outputs[t])
                # outputs[t] = torch.tanh(outputs[t])

        # return the last output_frames of the outputs
        outputs = outputs[-output_frames:]
        # print('222')
        # 5-th order tensor of size [batch_size, output_frames, channels, height, width]
        # for i in range(output_frames):
        #     outputs[i] = inputs[:, i].squeeze(1) + outputs[i]
        # outputs = inputs
        outputs = torch.stack([outputs[t] for t in range(output_frames)], dim=1)
        # print('222222222222')
        return outputs, input_2, input_3
