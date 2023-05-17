
import torch
import torch.nn as nn
import math

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
    def __init__(self, input_size, hidden_size, num_layers, num_heads, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.pos_encoder = PositionalEncoding(hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(hidden_size, num_heads, hidden_size, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = hidden_size
        
    def forward(self, input_ids, attention_mask, output_hidden_states=True):
        x = self.embedding(input_ids)
        x = self.pos_encoder(x)
        x = x.permute(1, 0, 2) # switch batch_size and sequence_length dimensions
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2) # switch back
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





