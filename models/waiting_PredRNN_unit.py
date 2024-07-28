import torch as t
from torch import nn


class predrnn_unit(nn.Module):
    def __init__(self, in_channel, hidden_channel, memory_channel):
        super(predrnn_unit, self).__init__()
        self.in_channel = in_channel
        self.hidden_channel = hidden_channel
        self.memory_channel = memory_channel
        self.conv_c = nn.Conv2d(self.in_channel + self.hidden_channel, self.hidden_channel * 3, 3, stride=1, padding=1)
        self.conv_m = nn.Conv2d(self.in_channel + self.memory_channel, self.memory_channel * 3, 3, stride=1, padding=1)
        self.conv_o = nn.Conv2d(self.in_channel + 2 * self.hidden_channel + self.memory_channel, self.hidden_channel, 3,
                                stride=1, padding=1)
        self.conv_h = nn.Conv2d(self.hidden_channel + self.memory_channel, self.hidden_channel, 1, stride=1, padding=0)

    def forward(self, x, state, m):
        h, c = state
        g, i, f = t.split(self.conv_c(t.cat((x, h), dim=1)), self.hidden_channel, dim=1)
        c = t.sigmoid(f) * c + t.sigmoid(i) * t.tanh(g)
        g_, i_, f_ = t.split(self.conv_m(t.cat((x, m), dim=1)), self.memory_channel, dim=1)
        m = t.sigmoid(f_) * m + t.sigmoid(i_) * t.tanh(g_)
        o = t.sigmoid(self.conv_o(t.cat((x, h, c, m), dim=1)))
        h = o * t.tanh(self.conv_h(t.cat((c, m), dim=1)))
        state = (h, c)
        return state, m
