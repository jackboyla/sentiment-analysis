
import torch
import torch.nn as nn

class SentimentLSTM(nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_size, num_layers, dropout=0.1):
        super(SentimentLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim, 
            hidden_size=self.hidden_size, 
            num_layers=self.num_layers, 
            dropout=dropout,
            batch_first=True
            )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_ids, attention_mask, output_hidden_states=True):
        embedded = self.embedding(input_ids)
        lstm_out, hidden = self.lstm(embedded)
        last_hidden_state = lstm_out[:, -1, :]

        return {'pooler_output': last_hidden_state}
    
