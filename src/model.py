import torch, torch.nn as nn
import torchmetrics
import lightning.pytorch as pl
import utils



class SentimentClassifier(pl.LightningModule):
    def __init__(self, tokenizer, hyperparams):
        super().__init__()
        
        self.cfg = hyperparams
        self.tokenizer = tokenizer

        self.lr = self.cfg.lr

        # Load Backbone
        self.encoder = utils.load_obj(self.cfg.backbone.object, name=self.cfg.backbone.name)

        # Freeze encoder if sepcified
        if self.cfg.freeze_encoder:
            total_params = sum(1 for _ in self.encoder.parameters())
            total_frozen_params = round(float(self.cfg.freeze_encoder) * total_params)
            i = 0
            # Freeze the weights of the encoder
            for param in self.encoder.parameters():
                if i < total_frozen_params:
                    param.requires_grad = False
                i += 1
            print(f"Frozen the first {total_frozen_params} out of {total_params} encoder weights")


        # Dropout
        self.dropout = nn.Dropout(self.encoder.config.hidden_dropout_prob) # 0.1 for canine-c

        # Define Classifier Head
        self.hidden_size = self.encoder.config.hidden_size
        self.num_classes = self.cfg.num_classes
        self.classifier_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.num_classes)
        )

        # Loss + Metrics
        self.criterion = torch.nn.CrossEntropyLoss()
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes)
  
        self.train_f1 = torchmetrics.F1Score(task="multiclass", num_classes=self.num_classes)
        self.val_f1 = torchmetrics.F1Score(task="multiclass", num_classes=self.num_classes)

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
        self.train_acc(logits, labels)
        self.train_f1(logits, labels)
        self.log("train_loss", loss, prog_bar=True)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=False)
        self.log('train_F1', self.train_f1, on_step=True, on_epoch=False)
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        logits = self.get_logits(inputs)
        val_loss = self.criterion(logits, labels)
        self.val_acc(logits, labels)
        self.val_f1(logits, labels)
        self.log("val_loss", val_loss)
        self.log('val_acc', self.val_acc, on_step=True, on_epoch=True)
        self.log('val_F1', self.val_f1, on_step=True, on_epoch=True)

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        logits = self.get_logits(inputs)
        test_loss = self.criterion(logits, labels)
        self.log("test_loss", test_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
