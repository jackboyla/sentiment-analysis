
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
        
        self.dropout = nn.Dropout(cfg.dropout)
        
        
    def forward(self, input_ids, sorted_lengths, src_key_padding_mask=None, attention_mask=None, token_type_ids=None, output_hidden_states=True):

        B = input_ids.shape[0]
        x = self.embedding(input_ids)
        x = self.dropout(x)
        # x = pack_padded_sequence(x, sorted_lengths, batch_first=True) # unpad
        output, (hidden, _) = self.lstm(x)

        last_hidden_state = hidden[-1, :, :]
        last_hidden_state = self.dropout(last_hidden_state)

        return {'pooler_output': last_hidden_state}
    

class SentimentConvLSTM(nn.Module):
    def __init__(self, cfg):
        super(SentimentConvLSTM, self).__init__()

        self.cfg = cfg

        self.conv1d_filters = cfg.conv1d_filters
        self.conv1d_filters = cfg.conv1d_filters
        self.conv1d_kernel_size = cfg.conv1d_kernel_size
        self.max_pool_kernel_size = cfg.max_pool_kernel_size

        # self.lstm_input_size = int(self.conv1d_filters/self.max_pool_kernel_size)
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

        self.conv1d = nn.Conv1d(in_channels=self.embedding_dim,
                                out_channels=self.conv1d_filters, 
                                kernel_size=self.conv1d_kernel_size, 
                                padding='same')
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool1d(kernel_size=self.max_pool_kernel_size)

        self.lstm = nn.LSTM(input_size=self.conv1d_filters, 
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers, 
                            dropout=cfg.dropout,
                            bidirectional=self.bidirectional,
                            batch_first=True)
        
        self.dropout = nn.Dropout(cfg.dropout)
        
        
    def forward(self, input_ids, sorted_lengths, src_key_padding_mask=None, attention_mask=None, token_type_ids=None, output_hidden_states=True):

        B = input_ids.shape[0]                  # [B, seq_len]
        x = self.embedding(input_ids)           # [B, seq_len, embed_dim]
        x = self.dropout(x)
        x = x.permute(0, 2, 1)                  # [B, embed_dim, seq_len]
        x = self.conv1d(x)                      # [B, conv_filters, seq_len]
        x = self.relu(x)
        x = self.max_pool(x)                    # [B, conv_filters, (seq_len / pool_kernel)]
        x = x.permute(0, 2, 1)                  # [B, (seq_len / pool_kernel), conv_filters]
        x = self.dropout(x)
        output, (hidden, _) = self.lstm(x) 
        
        # Use the output of the last timestep for classification
        last_hidden_state = hidden[-1, :, :]
        last_hidden_state = self.dropout(last_hidden_state)

        return {'pooler_output': last_hidden_state}
    
