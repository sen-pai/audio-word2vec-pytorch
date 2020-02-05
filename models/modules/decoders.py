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
        # embedding_dim = config.get("embedding_dim", None)
        # self.embedding_dim = embedding_dim if embedding_dim is not None else self.hidden_size
        # self.embedding = nn.Embedding(config.get("n_classes", 32), self.embedding_dim, padding_idx=0)
        self.rnn = nn.GRU(
            # input_size=self.embedding_dim+self.hidden_size if config['decoder'].lower() == 'bahdanau' else self.embedding_dim,
            input_size = config['n_channels'],
            hidden_size=self.hidden_size,
            num_layers= self.num_layers,
            dropout=config.get("decoder_dropout", 0),
            bidirectional= self.bi,
            batch_first= False)
        # if config['decoder'] != "RNN":
        #     self.attention = Attention(
        #         self.batch_size,
        #         self.hidden_size,
        #         method=config.get("attention_score", "dot"),
        #         mlp=config.get("attention_mlp_pre", False))

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
        # print("input in decoder ", input.shape)
        # print("hidden in decoder", hidden.shape)

        #unsqueeze because input shape needed: (Sequence Length, Batch Size, Num Channels)
        # Here output only 1 per time step so Sequence Lenght = 1
        input = input.unsqueeze(0)

        if self.num_layers == 1:
        	if not self.bi:
	        	# print('entered')
	        	hidden = hidden.unsqueeze(0)

        # print("hidden",self.hidden_size)
        # print("batch_size", self.batch_size)
        # print("this is input size",input.size())
        # print("This is hidden size", hidden.size())

        # RNN (Eq 7 paper)
        # embedded = self.embedding(input).unsqueeze(0)
        # rnn_input = torch.cat((embedded, hidden.unsqueeze(0)), 2)  # NOTE : Tf concats `lambda inputs, attention: array_ops.concat([inputs, attention], -1)`.
        # rnn_output, rnn_hidden = self.rnn(rnn_input.transpose(1, 0), hidden.unsqueeze(0))
        rnn_output, rnn_hidden = self.rnn(input, hidden)
        # print("this is rnn_output", rnn_output)
        # print("this is rnn_output size", rnn_output.shape)
        output = rnn_output.squeeze(1)
        output = self.output_linear_layer(output)
        # print("final output", output)
        # print("final output shape", output.shape)

        # if self.decoder_output_fn:
        #     output = self.decoder_output_fn(output, -1)

        return output, rnn_hidden.squeeze(0)
