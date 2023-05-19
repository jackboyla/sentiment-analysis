
import torch
import torch.nn as nn
import utils

from transformers.models.canine.modeling_canine import CanineEmbeddings

class SentimentLSTM(nn.Module):
    def __init__(self, cfg):
        super(SentimentLSTM, self).__init__()

        self.cfg = cfg
        self.hidden_size = cfg.hidden_size
        self.embedding_dim = cfg.embedding_dim
        self.num_layers = cfg.num_layers

        if 'embedding' in cfg:
            self.embedding = utils.load_obj(cfg.embedding.object)(cfg)
        else:
            self.embedding = nn.Embedding(cfg.input_size, cfg.embedding_dim)
            
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim, 
            hidden_size=self.hidden_size, 
            num_layers=self.num_layers, 
            dropout=cfg.dropout,
            batch_first=True
            )
        self.dropout = nn.Dropout(cfg.dropout)
        
    def forward(self, input_ids, attention_mask, output_hidden_states=True):
        embedded = self.embedding(input_ids)
        lstm_out, hidden = self.lstm(embedded)
        last_hidden_state = lstm_out[:, -1, :]

        return {'pooler_output': last_hidden_state}
    
