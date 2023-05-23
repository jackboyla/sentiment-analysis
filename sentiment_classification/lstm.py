
import torch.nn as nn
import utils
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class SentimentLSTM(nn.Module):
    def __init__(self, cfg):
        super(SentimentLSTM, self).__init__()

        self.cfg = cfg
        self.bidirectional = cfg.bidirectional

        if self.bidirectional:
            self.bidirectional_scaling = 2
        else:
            self.bidirectional_scaling = 1
        self.hidden_size = cfg.hidden_size * self.bidirectional_scaling

        self.embedding_dim = cfg.embedding_dim
        self.num_layers = cfg.num_layers

        if 'embedding' in cfg:
            self.embedding = utils.load_obj(cfg.embedding.object)(cfg)
        else:
            self.embedding = nn.Embedding(cfg.input_size, cfg.embedding_dim, padding_idx=0)

        self.lstm = nn.LSTM(input_size=self.embedding_dim, 
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers, 
                            dropout=cfg.dropout,
                            bidirectional=self.bidirectional,
                            batch_first=True)
        
    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        h0 = torch.zeros((self.num_layers * self.bidirectional_scaling, batch_size, self.hidden_size))
        c0 = torch.zeros((self.num_layers * self.bidirectional_scaling, batch_size, self.hidden_size))
        hidden = (h0,c0)
        return hidden
        
    def forward(self, input_ids, sorted_lengths, src_key_padding_mask=None, attention_mask=None, token_type_ids=None, output_hidden_states=True):

        B = input_ids.shape[0]
        x = self.embedding(input_ids)
        x = pack_padded_sequence(x, sorted_lengths, batch_first=True) # unpad
        self.hidden = self.init_hidden(B)
        output, (self.hidden, _) = self.lstm(x, self.hidden)
        # output, lengths = pad_packed_sequence(output)
        last_hidden_state = self.hidden[-1]
        # last_hidden_state = self.hidden.squeeze()

        return {'pooler_output': last_hidden_state}
    
