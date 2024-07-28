import torch as t
from torch import nn
from .waiting_PredRNN_unit import predrnn_unit


class predrnn_layer(nn.Module):
    def __init__(self, in_channel, hidden_channels, memory_channel, use_gpu=True):
        super(predrnn_layer, self).__init__()
        self.in_channel = in_channel
        self.hidden_channels = hidden_channels
        self.memory_channel = memory_channel
        self.layer_number = len(self.hidden_channels)
        self.use_gpu = use_gpu
        predrnn_units_list = []
        for i in range(self.layer_number):
            cur_in_channel = self.in_channel if i == 0 else self.hidden_channels[i - 1]
            cur_hidden_channel = self.hidden_channels if self.layer_number == 1 else self.hidden_channels[i]
            predrnn_units_list.append(predrnn_unit(cur_in_channel, cur_hidden_channel, self.memory_channel))
        self.predrnn_units_list = nn.ModuleList(predrnn_units_list)

    def zero_ini_layers_states(self, batch_size, height, width):
        ini_layers_states = []
        for i in range(self.layer_number):
            cur_hidden_channel = self.hidden_channels if self.layer_number == 1 else self.hidden_channels[i]
            zero_state = t.zeros([batch_size, cur_hidden_channel, height, width])
            if self.use_gpu:
                zero_state = zero_state.cuda()
            zero_layer_states = (zero_state, zero_state)
            ini_layers_states.append(zero_layer_states)
        return ini_layers_states

    def zero_ini_memory(self, batch_size, height, width):
        zero_memory = t.zeros([batch_size, self.memory_channel, height, width])
        if self.use_gpu:
            zero_memory = zero_memory.cuda()
        return zero_memory

    def forward(self, input, ini_layers_states=None, ini_memory=None):
        '''

        :param input: (batch_size, in_channel, height, width)
        :param ini_layers_states: [(h1, c1), (h2, c2),...]
        :param ini_memory: (batch_size, memory_channel, height, width)
        :return:
        '''
        batch_size, channel, height, width = input.size()
        if ini_layers_states is None:
            ini_layers_states = self.zero_ini_layers_states(batch_size, height, width)
        if ini_memory is None:
            ini_memory = self.zero_ini_memory(batch_size, height, width)
        cur_input = input
        cur_memory = ini_memory
        layers_states = []
        for layer_index in range(self.layer_number):
            state = ini_layers_states[layer_index]
            state, cur_memory = self.predrnn_units_list[layer_index](cur_input, state, cur_memory)
            layers_states.append(state)
            cur_input = state[0]
        return cur_input, layers_states, cur_memory
