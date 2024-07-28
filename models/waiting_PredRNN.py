import torch as t
from torch import nn
from einops import rearrange
import torch.nn.functional as F
from .PredRNN_layer import predrnn_layer


class FeatureEmbedding(nn.Module):
    def __init__(self, hidden_channels):
        super(FeatureEmbedding, self).__init__()
        self.hidden_channels = hidden_channels
        self.conv1 = nn.Conv2d(1, self.hidden_channels[0], 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(self.hidden_channels[0], self.hidden_channels[1], 3, stride=2, padding=1)

    def forward(self, x):
        '''

        :param x (batch_size, seq_len, 1, height, width)
        :return: x (batch_size, seq_len, channel, height / 8, width / 8)
        '''
        batch_size, seq_len, channel, height, width = x.size()
        x = rearrange(x, 'b l c h w -> (b l) c h w')
        x = F.interpolate(x, [int(height / 2), int(width / 2)], mode='bilinear')
        x = F.leaky_relu(self.conv1(x), negative_slope=0.1, inplace=True)
        x = rearrange(F.leaky_relu(self.conv2(x), negative_slope=0.1, inplace=True), '(b l) c h w -> b l c h w',
                      b=batch_size, l=seq_len)
        return x


class FeatureDecoding(nn.Module):
    def __init__(self, hidden_channels):
        super(FeatureDecoding, self).__init__()
        self.hidden_channels = hidden_channels
        self.deconv1 = nn.ConvTranspose2d(self.hidden_channels[0], self.hidden_channels[1], 2, stride=2, padding=0)
        self.deconv2 = nn.ConvTranspose2d(self.hidden_channels[1], self.hidden_channels[2], 2, stride=2, padding=0)
        self.conv = nn.Conv2d(self.hidden_channels[2], 1, 1, stride=1, padding=0)

    def forward(self, x):
        '''

        :param x (batch_size, seq_len, channel, height / 8, width / 8)
        :return: x (batch_size, seq_len, 1, height, width)
        '''
        batch_size, seq_len, channel, height, width = x.size()
        x = rearrange(x, 'b l c h w -> (b l) c h w')
        x = F.leaky_relu(self.deconv1(x), negative_slope=0.1, inplace=True)
        x = F.leaky_relu(self.deconv2(x), negative_slope=0.1, inplace=True)
        x = t.sigmoid(self.conv(x))
        x = rearrange(F.interpolate(x, [int(height * 8), int(width * 8)], mode='bilinear'), '(b l) c h w -> b l c h w',
                      b=batch_size, l=seq_len)
        return x


class PredRNN(nn.Module):
    def __init__(self):
        super(PredRNN, self).__init__()
        self.in_len = 10
        self.out_len = 10
        self.FE_channels = [32, 64]
        self.PR_hidden_channels = [64, 64, 64, 64]
        self.FD_channels = [64, 32, 16]
        self.memory_channel = 64
        self.coefficient = 0.00002
        self.use_gpu = True
        self.feature_embedding = FeatureEmbedding(self.FE_channels)
        self.predrnn = predrnn_layer(self.FE_channels[-1], self.PR_hidden_channels, self.memory_channel, self.use_gpu)
        self.feature_decoding = FeatureDecoding(self.FD_channels)

    def forward(self, x, in_len, out_len, truth=None, iter=None):
        '''

        :param x: (batch_size, in_len, 1, height, width)
        :param truth: (batch_size, out_len, 1, height, width)
        :param iter:
        :return: output: (batch_size, in_len + out_len - 1, 1, height, width)
        '''
        x = self.feature_embedding(x)
        if truth is not None:
            truth = self.feature_embedding(truth)
        layer_states = None
        memory = None
        output = []
        # for i in range(self.in_len):
        for i in range(in_len):
            input = x[:, i]
            hidden_state, layer_states, memory = self.predrnn(input, layer_states, memory)
            output.append(hidden_state)
        # for j in range(self.out_len - 1):
        for j in range(out_len - 1):
            if truth is None:
                input = hidden_state
            else:
                input = min(self.coefficient * iter * hidden_state, 1) + max((1 - self.coefficient * iter), 0) * truth[
                                                                                                                 :, j]
            hidden_state, layer_states, memory = self.predrnn(input, layer_states, memory)
            output.append(hidden_state)
        output = t.stack(output, dim=1)
        output = self.feature_decoding(output)
        # x = output
        return output
