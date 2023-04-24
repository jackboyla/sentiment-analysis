# import torch.nn as nn

# class SentimentClassifier(nn.Module):
#     def __init__(self, hidden_size, num_classes):
#         super(SentimentClassifier, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_classes = num_classes
#         self.classifier = nn.Sequential(
#             nn.Linear(hidden_size, hidden_size),
#             nn.ReLU(),
#             nn.Linear(hidden_size, num_classes)
#         )

#     def forward(self, inputs):
#         # get last hidden state of the Reformer
#         hidden_states = inputs.hidden_states[-1]
#         pooled_output = hidden_states[:, 0]
#         logits = self.classifier(pooled_output)
#         return logits
    
import torch, torch.nn as nn
import pytorch_lightning as L
from transformers import ReformerModelWithLMHead

# --------------------------------
# Step 1: Define a LightningModule
# --------------------------------
# A LightningModule (nn.Module subclass) defines a full *system*
# (ie: an LLM, diffusion model, autoencoder, or simple image classifier).


class SentimentClassifier(L.LightningModule):
    def __init__(self, tokenizer):
        super().__init__()
        self.reformer = ReformerModelWithLMHead.from_pretrained("google/reformer-enwik8")
        # # Freeze the weights of the first layer (fc1)
        # for param in self.reformer.parameters():
        #     param.requires_grad = False

        self.hidden_size = self.reformer.config.hidden_size
        self.num_classes = 2
        self.classifier_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.num_classes)
        )

        self.tokenizer = tokenizer
        self.criterion = torch.nn.CrossEntropyLoss()

    def get_logits(self, inputs):
        reformer_output = self.reformer(**inputs, output_hidden_states=True)  
        pooled_output = self.get_last_hidden_state_output(reformer_output)
        logits = self.classifier_head(pooled_output)
        return logits

    def get_last_hidden_state_output(self, reformer_output):
        # get last hidden state of the Reformer
        hidden_states = reformer_output.hidden_states[-1]
        pooled_output = hidden_states[:, 0]
        return pooled_output

    def forward(self, text):
        # in lightning, forward defines the prediction/inference actions

        # Tokenize input text
        # text = "I didn't think sheep were going to be so wonderful! :)"
        input_ids, attention_masks = self.tokenizer.encode([text])

        # Make prediction
        reformer_logits = self.reformer(**{'input_ids': input_ids}, output_hidden_states=True)  
        pooled_output = self.get_last_hidden_state_output(reformer_logits)
        logits = self.classifier_head(pooled_output)
        preds = torch.argmax(logits, dim=1).flatten().tolist()
        return preds

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        inputs, labels = batch
        # outputs : torch.Size([B, seq_len, out_features=258])
        # hidden_states : list of len 13 each of torch.Size([B, seq_len, hidden_size=1024])
        logits = self.get_logits(inputs)
        loss = self.criterion(logits, labels)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        logits = self.get_logits(inputs)
        val_loss = self.criterion(logits, labels)
        self.log("val_loss", val_loss)

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        logits = self.get_logits(inputs)
        test_loss = self.criterion(logits, labels)
        self.log("test_loss", test_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        return optimizer
