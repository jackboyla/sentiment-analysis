import torch, torch.nn as nn
import pytorch_lightning as L
from transformers import CanineModel

class SentimentClassifier(L.LightningModule):
    def __init__(self, tokenizer, freeze_encoder=True):
        super().__init__()
        # self.encoder = ReformerModelWithLMHead.from_pretrained("google/reformer-enwik8")
        self.encoder = CanineModel.from_pretrained("google/canine-c")
        self.encoder = self.encoder

        if freeze_encoder:
            # Freeze the weights of the first layer (fc1)
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(self.encoder.config.hidden_dropout_prob)

        self.hidden_size = self.encoder.config.hidden_size
        self.num_classes = 2
        self.classifier_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.num_classes)
        )

        self.tokenizer = tokenizer
        self.criterion = torch.nn.CrossEntropyLoss()

    def get_logits(self, inputs):
        encoder_output = self.encoder(**inputs, output_hidden_states=True) 
        pooled_output = encoder_output['pooler_output']
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier_head(pooled_output)
        return logits

    def forward(self, text):
        # in lightning, forward defines the prediction/inference actions

        # Tokenize input text
        # text = "I didn't think sheep were going to be so wonderful! :)"
        inputs = self.tokenizer(text, return_tensors='pt')

        # Make prediction
        logits = self.get_logits(inputs)
        preds = torch.argmax(logits, dim=1).flatten().tolist()
        return preds

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        inputs, labels = batch
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
