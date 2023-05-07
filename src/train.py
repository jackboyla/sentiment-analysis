

import importlib
import sys
import os
from torch.utils.data import DataLoader
import dataflow as dataflow
import model as models
import utils

import lightning as L

from sklearn.model_selection import train_test_split

import pandas as pd
from omegaconf import OmegaConf
from transformers import CanineTokenizer

from lightning.pytorch.cli import SaveConfigCallback
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, ModelSummary, DeviceStatsMonitor
from lightning.pytorch.loggers import WandbLogger, CSVLogger
import time
import wandb

cfg_path = sys.argv[1]  # 'configs/local-canine-backbone.yaml'
cfg = OmegaConf.load(cfg_path)

server_log_file = sys.argv[2]

DATASET_COLUMNS = ["target", "ids", "date", "flag", "user", "text"]
df = pd.read_csv(
    cfg.datafiles.data_dir,
    encoding=cfg.datafiles.dataset_encoding,
    names=DATASET_COLUMNS
    )

decode_map = {0: 0, 4: 1}
def binarize_sentiment(label):
    return decode_map[int(label)]

df['target'] = df.target.apply(lambda x: binarize_sentiment(x))

train_size = int(cfg.data_processing.train_size if cfg.data_processing.train_size > 1.0 else cfg.data_processing.train_size*len(df))
val_size = int(cfg.data_processing.val_size if cfg.data_processing.val_size > 1.0 else cfg.data_processing.val_size*len(df))
test_size = int(cfg.data_processing.test_size if cfg.data_processing.test_size > 1.0 else cfg.data_processing.test_size*len(df))

train_df, test_df = train_test_split(df, train_size=train_size+val_size, test_size=test_size, random_state=42, stratify=df['target'])
train_df, val_df = train_test_split(train_df, train_size=train_size, test_size=val_size, random_state=42, stratify=train_df['target'])

train_df.reset_index(inplace=True, drop=True)
val_df.reset_index(inplace=True, drop=True)
test_df.reset_index(inplace=True, drop=True)



# single sequence: [CLS] X [SEP]
tokenizer = CanineTokenizer.from_pretrained("google/canine-c", 
                                            return_tensors='pt')

# Build Datasets
train_dataset = dataflow.TweetDataset(train_df['text'], train_df['target'], tokenizer)
val_dataset = dataflow.TweetDataset(val_df['text'], val_df['target'], tokenizer)
test_dataset = dataflow.TweetDataset(test_df['text'], test_df['target'], tokenizer)


# Build DataLoaders
train_dataloader = DataLoader(train_dataset, 
                              batch_size=cfg.hyperparameters.batch_size, 
                              shuffle=True, 
                              collate_fn=lambda b: dataflow.collate_fn(b, input_pad_token_id=tokenizer.pad_token_id)
                              )
val_dataloader = DataLoader(val_dataset, batch_size=cfg.hyperparameters.batch_size, shuffle=False, collate_fn=lambda b: dataflow.collate_fn(b, input_pad_token_id=tokenizer.pad_token_id))
test_dataloader = DataLoader(test_dataset, batch_size=cfg.hyperparameters.batch_size, shuffle=False, collate_fn=lambda b: dataflow.collate_fn(b, input_pad_token_id=tokenizer.pad_token_id))


if cfg.logging.wandb:
    # # DO NOT login if you want to log offline
    # wandb.login(key=os.environ['WANDB_API_KEY'])
    wandb_logger = WandbLogger(project=cfg.logging.project_name, 
                               offline=True
                               )
    
slack_callback = utils.SlackCallback(webhook_url=os.environ['SLACK_HOOK'], 
                                     cfg=OmegaConf.to_yaml(cfg),
                                     server_log_file=server_log_file
                                     )

log_name = 'lightning_logs'
version = time.strftime("%Y-%m-%d__%H-%M")
csv_logger = CSVLogger(cfg.logging.log_dir, name=log_name, version=version)
# save_config_callback = SaveConfigCallback(config=cfg.logging.log_dir)

# with open(os.path.join(*[cfg.logging.log_dir, log_name, version, 'config.yaml']), 'w') as f:
#     OmegaConf.save(config=cfg, f=f)

print(f"Training on {len(train_df)} examples...")

early_stop = EarlyStopping(cfg.callbacks.early_stopping.monitor, 
                           patience=cfg.callbacks.early_stopping.patience, 
                           verbose=True, 
                           min_delta=cfg.callbacks.early_stopping.min_delta)

checkpoint_callback = ModelCheckpoint(save_top_k=cfg.callbacks.checkpoint.save_top_k, 
                                      monitor=cfg.callbacks.checkpoint.monitor)

trainer = L.Trainer(max_epochs=cfg.hyperparameters.max_epochs, 
                    profiler=cfg.hyperparameters.profiler,
                    log_every_n_steps=100,
                    logger=[wandb_logger, csv_logger],
                    enable_progress_bar=False,
                    callbacks=[ 
                        early_stop, checkpoint_callback, 
                        DeviceStatsMonitor(), 
                        utils.PrintTableMetricsCallback(),
                        slack_callback 
                               ], # 
                    )

classifier = models.SentimentClassifier(tokenizer=tokenizer, 
                                        hyperparams=cfg.hyperparameters)

# log gradients and model topology
wandb_logger.watch(classifier)

trainer.fit(classifier, train_dataloader, val_dataloader)

wandb_logger.experiment.unwatch(classifier)

trainer.test(classifier, dataloaders=test_dataloader)

if cfg.logging.wandb:
    # [optional] finish the wandb run, necessary in notebooks
    wandb.finish()