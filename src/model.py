import torch, torch.nn as nn
import torchmetrics
import lightning.pytorch as pl
import utils
import transformers


class SentimentClassifier(pl.LightningModule):
    def __init__(self, tokenizer, hyperparams):
        super().__init__()
        
        self.cfg = hyperparams
        self.tokenizer = tokenizer

        self.backbone_lr = self.cfg.optimizer.lr.backbone
        self.head_lr = self.cfg.optimizer.lr.head

        # Load Backbone
        if 'transformers' in self.cfg.backbone.object:
            self.encoder = transformers.AutoModel.from_pretrained(**self.cfg.backbone.kwargs)
            self.hidden_size = self.encoder.config.hidden_size
        else:
            self.encoder = utils.load_obj(self.cfg.backbone.object)
            self.encoder = self.encoder(input_size=self.tokenizer.vocab_size, **self.cfg.backbone.get('kwargs', {}))
            self.hidden_size = self.encoder.hidden_size

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
        self.dropout = nn.Dropout(0.1) # 0.1 for canine-c

        # Define Classifier Head
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
        self.test_f1 = torchmetrics.F1Score(task="multiclass", num_classes=self.num_classes)


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
        self.log("train_loss", loss)
        self.log('train_acc', self.train_acc)
        self.log('train_F1', self.train_f1)
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        logits = self.get_logits(inputs)
        val_loss = self.criterion(logits, labels)
        self.val_acc(logits, labels)
        self.val_f1(logits, labels)
        self.log("val_loss", val_loss)
        self.log('val_acc', self.val_acc)
        self.log('val_F1', self.val_f1)

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        logits = self.get_logits(inputs)
        test_loss = self.criterion(logits, labels)
        self.test_f1(logits, labels)
        self.log("test_loss", test_loss)
        self.log('test_F1', self.test_f1)

    def configure_optimizers(self):
        # https://github.com/Lightning-AI/lightning/issues/2005
        grouped_parameters = [
            {"params": [p for p in self.encoder.parameters() if p.requires_grad], 
             'lr': self.backbone_lr,
             'name': 'backbone_LR'
             },

            {"params": [p for p in self.classifier_head.parameters()], 
             'lr': self.head_lr,
             'name': 'head_LR'
             },
        ]

        optimizer = utils.load_obj(self.cfg.optimizer.object)
        optimizer = optimizer(grouped_parameters, **self.cfg.optimizer.get('kwargs', {}))

        scheduler = utils.load_obj(self.cfg.scheduler.object)
        scheduler = scheduler(optimizer,
                              num_training_steps=self.trainer.estimated_stepping_batches, 
                              **self.cfg.scheduler.get('kwargs', {}))
        # scheduler = transformers.get_linear_schedule_with_warmup(
        #     optimizer,
        #     num_warmup_steps=0,
        #     num_training_steps=self.trainer.estimated_stepping_batches,
        # )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]
