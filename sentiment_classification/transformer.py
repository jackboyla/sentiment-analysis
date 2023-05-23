
import torch
import torch.nn as nn
import math
import utils

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, dropout=0.1, max_seq_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.embedding_dim = embedding_dim

        # Initialize the positional encoding matrix
        pe = torch.zeros(max_seq_len, embedding_dim)

        # Compute the positional encoding values for each position and dimension
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add the positional encoding matrix as a buffer parameter
        # The buffer parameter won't be updated during backpropagation
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add the positional encoding to the input embeddings
        x = x * math.sqrt(self.embedding_dim)
        seq_len = x.size(1)
        pe = self.pe[:seq_len, :]
        pe = pe.unsqueeze(0).repeat(x.size(0), 1, 1)
        x = x + pe
        x = self.dropout(x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, cfg):
        super(TransformerEncoder, self).__init__()
        
        self.hidden_size = cfg.hidden_size
        self.input_size = cfg.input_size
        self.embedding_dim = cfg.embedding_dim
        self.dropout = cfg.dropout
        self.num_heads = cfg.num_heads
        self.num_layers = cfg.num_layers
        self.cfg = cfg

        if 'embedding' in cfg:
            self.embedding = utils.load_obj(self.cfg.embedding.object)(self.cfg)
        else:
            self.embedding = nn.Embedding(num_embeddings=self.input_size, 
                                          embedding_dim=self.embedding_dim)

        self.pos_encoder = PositionalEncoding(self.hidden_size, self.dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size, 
            nhead=self.num_heads, 
            dropout=self.dropout, 
            batch_first=True
            )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, self.num_layers)
        
    def forward(self, input_ids, attention_mask, output_hidden_states=True):
        x = self.embedding(input_ids)  # [B, seq_len, hidden_dim]
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        return {'pooler_output': x}





