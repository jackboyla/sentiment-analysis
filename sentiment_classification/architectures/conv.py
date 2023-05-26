
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
        
        pooled = [F.max_pool1d(input=conv, kernel_size=conv.shape[2]).squeeze(2) for conv in conved]
        # pooled[i] shape: (batch_size, num_filters)
        
        cat = torch.cat(pooled, dim=1)
        # cat shape: (batch_size, num_filters * len(kernel_sizes))
        
        # Apply dropout
        pooler_output = self.dropout(cat)
        
        return {'pooler_output': pooler_output}
    
