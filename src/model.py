import torch.nn as nn

class SentimentClassifier(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super(SentimentClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, inputs):
        # get last hidden state of the Reformer
        hidden_states = inputs.hidden_states[-1]
        pooled_output = hidden_states[:, 0]
        logits = self.classifier(pooled_output)
        return logits
    