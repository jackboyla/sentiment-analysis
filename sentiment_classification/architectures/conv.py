
import torch.nn as nn
import utils
import torch
import torch.nn.functional as F

    
class SentimentCNN(nn.Module):
    def __init__(self, cfg):
        super(SentimentCNN, self).__init__()

        self.cfg = cfg
        self.input_size = cfg.input_size
        self.embedding_dim = cfg.embedding_dim
        self.num_filters = cfg.num_filters
        self.kernel_sizes = cfg.kernel_sizes

        if 'embedding' in cfg:
            self.embedding = utils.load_obj(cfg.embedding.object)(cfg)
        else:
            self.embedding = nn.Embedding(self.input_size, self.embedding_dim, padding_idx=0)


        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(cfg.dropout)

        # Convolutional layers
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=self.embedding_dim, out_channels=self.num_filters, kernel_size=fs) 
            for fs in self.kernel_sizes
        ])

        self.hidden_size = len(self.kernel_sizes) * self.num_filters
        
        
    def forward(self, input_ids, sorted_lengths=None, src_key_padding_mask=None, attention_mask=None, token_type_ids=None, output_hidden_states=True):
        embedded = self.embedding(input_ids)
        embedded = embedded.permute(0, 2, 1)
        embedded = self.dropout(embedded)
        # embedded shape: (batch_size, embedding_dim, seq_len)
        
        conved = [self.relu(conv(embedded)) for conv in self.convs]
        # conved[i] shape: (batch_size, num_filters, seq_len - filter_sizes[i] + 1)
        
        pooled = [torch.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        # pooled[i] shape: (batch_size, num_filters)
        
        cat = torch.cat(pooled, dim=1)
        # cat shape: (batch_size, num_filters * len(kernel_sizes))
        
        # Apply dropout
        pooler_output = self.dropout(cat)
        
        return {'pooler_output': pooler_output}
    


class Sentiment2dCNN(nn.Module):
    def __init__(self, cfg):
        super(Sentiment2dCNN, self).__init__()

        self.cfg = cfg

        self.input_size = cfg.input_size
        self.embedding_dim = cfg.embedding_dim
        self.num_filters = cfg.num_filters
        self.kernel_sizes = cfg.kernel_sizes

        self.dropout = cfg.dropout

        if 'embedding' in cfg:
            self.embedding = utils.load_obj(cfg.embedding.object)(cfg)
        else:
            self.embedding = nn.Embedding(self.input_size, self.embedding_dim, padding_idx=0)

        self.hidden_size = len(self.kernel_sizes) * self.num_filters


        # Convolutional layers
        self.convs = nn.ModuleList([
            nn.Conv2d(1, self.num_filters, (fs, self.embedding_dim)) for fs in self.kernel_sizes
        ])

        # Dropout
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, input_ids, sorted_lengths=None, src_key_padding_mask=None, attention_mask=None, token_type_ids=None, output_hidden_states=True):
        x = self.embedding(input_ids)  # [B, seq_len, embed_dim]

        # Add a channel dimension for conv2d
        x = x.unsqueeze(1) # [B, 1, seq_len, embed_dim]

        # Apply convolutional layers
        conved = [torch.relu(conv(x)).squeeze(3) for conv in self.convs] # [ [B, num_filters, seq_len], .... ]

        # Apply max pooling over time
        pooled = [torch.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved] # [ [B, num_filters], ... ]

        # Concatenate pooled outputs
        cat = torch.cat(pooled, dim=1) # [B, len(self.kernel_sizes) * self.num_filters]

        # Apply dropout
        pooler_output = self.dropout(cat)

        return {'pooler_output': pooler_output}
    
