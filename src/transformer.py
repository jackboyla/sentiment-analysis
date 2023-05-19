
import torch
import torch.nn as nn
import math
import utils

from transformers.models.canine.modeling_canine import CanineEmbeddings

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_seq_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=0.1)
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
        self.cfg = cfg

        if 'embedding' in cfg:
            self.embedding = utils.load_obj(cfg.embedding.object)(cfg)
        else:
            self.embedding = nn.Embedding(cfg.input_size, cfg.embedding_dim)

        self.pos_encoder = PositionalEncoding(cfg.hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            cfg.hidden_size, cfg.num_heads, 
            cfg.hidden_size, cfg.dropout, 
            batch_first=True
            )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, cfg.num_layers)
        self.dropout = nn.Dropout(cfg.dropout)
        self.hidden_size = cfg.hidden_size
        
    def forward(self, input_ids, attention_mask, output_hidden_states=True):
        x = self.embedding(input_ids)  # [B, seq_len, hidden_dim]
        # x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return {'pooler_output': x}


        
    # def forward(self, x):
    #     embedded = self.embedding(x)
    #     embedded = embedded.permute(1, 0, 2) # switch batch_size and sequence_length dimensions
    #     encoded = self.transformer_encoder(embedded)
    #     encoded = encoded.permute(1, 0, 2) # switch back
    #     pooled = F.avg_pool1d(encoded.transpose(1,2), encoded.shape[1]).squeeze(2) # average pooling across sequence length
    #     x = self.dropout(pooled)
    #     x = F.relu(self.fc1(x))
    #     x = self.dropout(x)
    #     x = self.fc2(x)
    #     return x





