import torch
import torch.nn as nn
import torch.nn.functional as F

from models.helpers import mask_3d


class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.batch_size = config["batch_size"]
        self.hidden_size = config["decoder_hidden"]
        self.bi = config.get("bidirectional_decoder", False)
        self.num_layers = config.get("decoder_layers", 1)
        
        # self.rnn = nn.GRU(
        #     input_size = config['n_channels'],
        #     hidden_size=self.hidden_size,
        #     num_layers= self.num_layers,
        #     dropout=config.get("decoder_dropout", 0),
        #     bidirectional= self.bi,
        #     batch_first= False)
        
        self.rnn = nn.LSTM(
            input_size = config['n_channels'],
            hidden_size=self.hidden_size,
            num_layers= self.num_layers,
            dropout=config.get("decoder_dropout", 0),
            bidirectional= self.bi,
            batch_first= False)

        self.gpu = config.get("gpu", False)
        # self.decoder_output_fn = F.log_softmax if config.get('loss', 'NLL') == 'NLL' else None

    def forward(self, **kwargs):
        """ Must be overrided """
        raise NotImplementedError


class RNNDecoder(Decoder):
    def __init__(self, config):
        super(RNNDecoder, self).__init__(config)
        # self.output_size = config.get("n_classes", 32)
        #shape of output tensor
        self.output_size = config['n_channels']
        if self.bi:
        	hidden_bi_size = self.hidden_size*2
        else:
        	hidden_bi_size = self.hidden_size
        self.output_linear_layer = nn.Linear(hidden_bi_size, self.output_size)

    def forward(self, **kwargs):
        input = kwargs["input"]
        hidden = kwargs["hidden"]
        cell = kwargs["cell"]
        # print("input in decoder ", input.shape)
        # print("hidden in decoder", hidden.shape)

        #unsqueeze because input shape needed: (Sequence Length, Batch Size, Num Channels)
        # Here output only 1 per time step so Sequence Lenght = 1
        input = input.unsqueeze(0)

        if self.num_layers == 1:
        	if not self.bi:
	        	# print('entered')
	        	hidden = hidden.unsqueeze(0)
	        	cell = cell.unsqueeze(0)

        rnn_output, (rnn_hidden, rnn_cell) = self.rnn(input, (hidden, cell))

        output = rnn_output.squeeze(1)
        output = self.output_linear_layer(output)

        # if self.decoder_output_fn:
        #     output = self.decoder_output_fn(output, -1)

        return output, rnn_hidden.squeeze(0), rnn_cell.squeeze(0)
