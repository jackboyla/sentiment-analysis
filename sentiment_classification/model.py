import torch, torch.nn as nn
import torchmetrics
import lightning.pytorch as pl
import utils
import transformers
import inspect

run_logger = utils.create_logger(__name__)

class SentimentClassifier(pl.LightningModule):
    def __init__(self, tokenizer, hyperparams):
        super().__init__()
        
        self.cfg = hyperparams
        self.tokenizer = tokenizer
        self.required_args = self.cfg.backbone.required_args

        self.backbone_lr = self.cfg.optimizer.lr.backbone
        self.head_lr = self.cfg.optimizer.lr.head

        # Load Backbone
        if 'transformers' in self.cfg.backbone.object:
            self.encoder = transformers.AutoModel.from_pretrained(self.cfg.backbone.kwargs.pretrained_model_name_or_path)
            self.hidden_size = self.encoder.config.hidden_size
        else:
            self.encoder = utils.load_obj(self.cfg.backbone.object)
            self.encoder = self.encoder(self.cfg.backbone.get('kwargs', {}))
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
            run_logger.info(f"Frozen the first {total_frozen_params} out of {total_params} encoder weights")
        else:
            run_logger.info(f"No encoder weights frozen")

        # Dropout
        self.dropout = nn.Dropout(0.1)

        # Define Classifier Head
        self.num_classes = self.cfg.num_classes
        self.classifier_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.num_classes),
            nn.LogSoftmax(dim=1)
            )
        
        # # Initialization
        # for m in [self.encoder, self.classifier_head]:
        #     self.init_weights(m)

        # Loss + Metrics
        # self.criterion = torch.nn.CrossEntropyLoss()
        self.criterion = torch.nn.NLLLoss()

        self.train_acc = torchmetrics.classification.MulticlassAccuracy(num_classes=self.num_classes, average='micro')
        self.val_acc = torchmetrics.classification.MulticlassAccuracy(num_classes=self.num_classes, average='micro')
  
        self.train_f1 = torchmetrics.classification.MulticlassF1Score(num_classes=self.num_classes, average='macro')
        self.val_f1 = torchmetrics.classification.MulticlassF1Score(num_classes=self.num_classes, average='macro')
        self.test_f1 = torchmetrics.classification.MulticlassF1Score(num_classes=self.num_classes, average='macro')


    def get_logits(self, inputs):
        inputs = {key: value for key, value in inputs.items() if key in self.required_args}
        encoder_output = self.encoder(**inputs, output_hidden_states=True) 
        pooled_output = encoder_output['pooler_output']  # [B, hidden_size]
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
        self.log("test_loss", test_loss, on_step=True)
        self.log('test_F1', self.test_f1)

    def configure_optimizers(self):
        # https://github.com/Lightning-AI/lightning/issues/2005#issuecomment-636218469
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
        self.grouped_parameters = grouped_parameters

        optimizer = utils.load_obj(self.cfg.optimizer.object)
        optimizer = optimizer(grouped_parameters, **self.cfg.optimizer.get('kwargs', {}))

        if 'scheduler' in self.cfg:

            scheduler = utils.load_obj(self.cfg.scheduler.object)

            # Get the parameter names of the function
            scheduler_params = inspect.signature(scheduler).parameters

            # Extract the keyword arguments from the dictionary based on the function's parameter names
            scheduler_kwargs = {}
            scheduler_kwargs['num_training_steps'] = self.trainer.estimated_stepping_batches
            scheduler_kwargs['T_max'] = self.trainer.estimated_stepping_batches
            scheduler_kwargs.update(self.cfg.scheduler.kwargs)
            scheduler_kwargs = {param: scheduler_kwargs.get(param) for param in scheduler_params if param in scheduler_kwargs}

            
            scheduler = scheduler(optimizer,
                                  **scheduler_kwargs)

            scheduler = {"scheduler": scheduler, 
                         "interval": self.cfg.scheduler.kwargs.interval, 
                         "frequency": self.cfg.scheduler.kwargs.frequency}
            return [optimizer], [scheduler]
        
        return [optimizer]

