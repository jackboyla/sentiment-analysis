
import torch.nn as nn
import utils
import torch

class SentimentLSTM(nn.Module):
    def __init__(self, cfg):
        super(SentimentLSTM, self).__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(self.cfg.input_size, self.cfg.embedding_dim)
        self.rnn = nn.LSTM(
            input_size    = self.cfg.embedding_dim,
            hidden_size   = self.cfg.hidden_size,
            bidirectional = True, 
            dropout       = .3
        )
        self.dropout = nn.Dropout(.2)

        self.hidden_size = cfg.hidden_size * 2

    def forward(self, x):                                   
        x = self.emb(x)        
        x = torch.transpose(x, 0, 1) 
        output, (hidden, cell) = self.rnn(x)                                              
        hidden = torch.transpose(hidden, 0, 1)
        hidden = torch.cat((
            hidden[:, 0, :],
            hidden[:, 1, :]
            ),
            dim = -1
        )                                                   
        # hidden = self.dropout(hidden) 
        # x = self.lin(hidden)
        # x = torch.log_softmax(x, dim = 1)
        return {'pooler_output': hidden}

# class SentimentLSTM(nn.Module):
#     def __init__(self, cfg):
#         super(SentimentLSTM, self).__init__()

#         self.cfg = cfg


#         self.hidden_size = cfg.hidden_size
#         # self.hidden_size = 64



#         self.embedding_dim = cfg.embedding_dim
#         self.num_layers = cfg.num_layers

#         if 'embedding' in cfg:
#             self.embedding = utils.load_obj(cfg.embedding.object)(cfg)
#         else:
#             self.embedding = nn.Embedding(cfg.input_size, cfg.embedding_dim, padding_idx=0)

#         # self.conv1d = nn.Conv1d(cfg.embedding_dim, 32, kernel_size=3, padding=1)
#         # self.relu = nn.ReLU()
#         # self.maxpool1d = nn.MaxPool1d(kernel_size=2)
#         # self.bidirectional_lstm = nn.LSTM(input_size=32, hidden_size=32, bidirectional=True)
        
            
#         self.lstm = nn.LSTM(
#             input_size=self.embedding_dim, 
#             hidden_size=self.hidden_size, 
#             num_layers=self.num_layers, 
#             dropout=cfg.dropout,
#             batch_first=True
#             )
        
#     def forward(self, input_ids, attention_mask, output_hidden_states=True):
#         embedded = self.embedding(input_ids)
#         lstm_out, hidden = self.lstm(embedded)
#         last_hidden_state = lstm_out[:, -1, :]

#         # embedded = self.embedding(input_ids)
#         # embedded = embedded.permute(0, 2, 1)
#         # conv_out = self.conv1d(embedded)
#         # conv_out = self.relu(conv_out)
#         # pooled = self.maxpool1d(conv_out)
#         # pooled = pooled.permute(2, 0, 1)
#         # lstm_out, _ = self.bidirectional_lstm(pooled)
#         # last_hidden_state = lstm_out[-1, :, :]

#         return {'pooler_output': last_hidden_state}
    
