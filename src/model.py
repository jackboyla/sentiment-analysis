import torch, torch.nn as nn
import pytorch_lightning as L
from transformers import CanineModel

class SentimentClassifier(L.LightningModule):
    def __init__(self, tokenizer, freeze_encoder=True, lr=1e-5, num_classes=2):
        super().__init__()
        
        self.lr = lr
        self.encoder = CanineModel.from_pretrained("google/canine-c")
        self.encoder = self.encoder

        if freeze_encoder:
            total_params = sum(1 for _ in self.encoder.parameters())
            i = 0
            # Freeze the weights of the encoder
            for param in self.encoder.parameters():
                if i < round(float(freeze_encoder)) * total_params:
                    param.requires_grad = False
                i += 1
            print(f"Frozen the first {float(freeze_encoder)*100}% of encoder weights")

        self.dropout = nn.Dropout(self.encoder.config.hidden_dropout_prob) # 0.1 for canine-c

        self.hidden_size = self.encoder.config.hidden_size
        self.num_classes = num_classes
        self.classifier_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.num_classes)
        )

        self.tokenizer = tokenizer
        self.criterion = torch.nn.CrossEntropyLoss()

        # save hyper-parameters to self.hparamsm auto-logged by wandb
        self.save_hyperparameters()

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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
