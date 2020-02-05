import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
# from models.helpers import skip_add_pyramid


class EncoderRNN(nn.Module):
    def __init__(self, config):
        super(EncoderRNN, self).__init__()
        
        self.input_size = config["n_channels"]
        self.hidden_size = config["encoder_hidden"]
        self.layers = config.get("encoder_layers", 1)
        self.dropout = config.get("encoder_dropout", 0.)
        self.bi = config.get("bidirectional_encoder", False)
        
        # self.rnn = nn.GRU(
        #     self.input_size,
        #     self.hidden_size,
        #     self.layers,
        #     dropout=self.dropout,
        #     bidirectional=self.bi,
        #     batch_first=True)

        
        self.rnn = nn.LSTM(
            self.input_size,
            self.hidden_size,
            self.layers,
            dropout=self.dropout,
            bidirectional=self.bi,
            batch_first=True)

        self.gpu = config.get("gpu", False)


    def forward(self, inputs, hidden, cell, input_lengths):

        x = pack_padded_sequence(inputs, input_lengths, batch_first=True)
        
        output, (hidden_state, cell_state) = self.rnn(x, (hidden, cell))
        output, _ = pad_packed_sequence(output, batch_first= True, padding_value=0.)
        
        # if self.bi:
        #     output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]
        #     print('bi output', output.shape)
        return output, hidden_state, cell_state

    def init_hidden(self, batch_size):
        #init hidden for gru
        first_dim = self.layers
        if self.bi:
            first_dim = first_dim*2
        # print("first_dim", first_dim)

        h0 = Variable(torch.zeros(first_dim, batch_size, self.hidden_size))
        # print("h0 shape", h0.shape)
        if self.gpu:
            h0 = h0.cuda()
        return h0

    def init_hidden_lstm(self, batch_size):
        #init hidden for lstm
        first_dim = self.layers
        if self.bi:
            first_dim = first_dim*2
        # print("first_dim", first_dim)

        h0 = Variable(torch.zeros(first_dim, batch_size, self.hidden_size))
        c0 = Variable(torch.zeros(first_dim, batch_size, self.hidden_size))
        # print("h0 shape", h0.shape)
        if self.gpu:
            h0 = h0.cuda()
            c0 = c0.cuda()
        return h0, c0



