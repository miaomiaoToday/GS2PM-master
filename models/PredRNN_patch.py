import torch as t
from torch import nn
from einops import rearrange
from .waiting_PredRNN_layer import predrnn_layer


class FeatureDecoding(nn.Module):
    def __init__(self, hidden_channel):
        super(FeatureDecoding, self).__init__()
        self.hidden_channel = hidden_channel
        self.conv = nn.Conv2d(self.hidden_channel, 64, 3, stride=1, padding=1)

    def forward(self, x):
        '''

        :param x (batch_size, seq_len, channel, height, width)
        :return: x (batch_size, seq_len, 64, height, width)
        '''
        batch_size, seq_len, channel, height, width = x.size()
        x = rearrange(x, 'b l c h w -> (b l) c h w')
        x = rearrange(t.sigmoid(self.conv(x)), '(b l) c h w -> b l c h w', b=batch_size, l=seq_len)
        return x


class PredRNN_patch(nn.Module):
    def __init__(self):
        super(PredRNN_patch, self).__init__()
        self.in_len = 10
        self.out_len = 10
        self.FE_channels = [32, 64]
        self.PR_hidden_channels = [64, 64, 64, 64]
        self.memory_channel = 64
        self.use_gpu = True
        self.predrnn = predrnn_layer(self.FE_channels[-1], self.PR_hidden_channels, self.memory_channel, self.use_gpu)
        self.feature_decoding = FeatureDecoding(self.PR_hidden_channels[-1])

    def patch_division(self, x):
        b, l, c, h, w = x.size()
        patch_h = int(h / 8)
        patch_w = int(w / 8)
        x = x.reshape([b, l, c, 8, patch_h, 8, patch_w]).permute(0, 1, 2, 3, 5, 4, 6).reshape(
            [b, l, c * 8 * 8, patch_h, patch_w])
        return x

    def patch_combiation(self, x):
        b, l, c, h, w = x.size()
        x = x.reshape([b, l, 1, 8, 8, h, w]).permute(0, 1, 2, 3, 5, 4, 6).reshape([b, l, 1, 8 * h, 8 * w])
        return x

    def forward(self, x, in_len, out_len):
        '''

        :param x: (batch_size, in_len, 1, height, width)
        :return: output: (batch_size, in_len + out_len - 1, 1, height, width)
        '''
        x = self.patch_division(x)
        layer_states = None
        memory = None
        output = []
        for i in range(in_len):
            input = x[:, i]
            hidden_state, layer_states, memory = self.predrnn(input, layer_states, memory)
            output.append(hidden_state)
        for j in range(out_len - 1):
            input = hidden_state
            hidden_state, layer_states, memory = self.predrnn(input, layer_states, memory)
            output.append(hidden_state)
        output = t.stack(output, dim=1)
        output = self.feature_decoding(output)
        output = self.patch_combiation(output)
        return output
