import torch.nn as nn
import utils



class JustEmbeddingEncoder(nn.Module):
    def __init__(self, cfg):
        super(JustEmbeddingEncoder, self).__init__()
        self.cfg = cfg
        self.hidden_size = cfg.embedding_dim

        if 'embedding' in cfg:
            self.embedding = utils.load_obj(cfg.embedding.object)(cfg)
        else:
            self.embedding = nn.Embedding(cfg.input_size, cfg.embedding_dim)

        
    def forward(self, input_ids, attention_mask, output_hidden_states=True):
        x = self.embedding(input_ids)  # [B, seq_len, embedding_dim]
        x = x.mean(dim=1) # [B, embedding_dim]
        return {'pooler_output': x}