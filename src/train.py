

import importlib
import sys
import os
from torch.utils.data import DataLoader
import dataflow as dataflow
import model as models

import pytorch_lightning as L

from sklearn.model_selection import train_test_split

import pandas as pd
from omegaconf import OmegaConf
from transformers import CanineTokenizer

from lightning.pytorch.cli import SaveConfigCallback
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, ModelSummary, DeviceStatsMonitor
from pytorch_lightning.loggers import WandbLogger, CSVLogger
import time
import wandb

cfg_path = sys.argv[1]  # 'configs/local-canine-backbone.yaml'
cfg = OmegaConf.load(cfg_path)

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

train_df, test_df = train_test_split(df, train_size=50000, test_size=10000, random_state=42, stratify=df['target'])
train_df, val_df = train_test_split(train_df, test_size=0.3, random_state=42, stratify=train_df['target'])

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
    wandb.login(key=os.environ['WANDB_API_KEY'])
    wandb_logger = WandbLogger(project=cfg.logging.project_name)
    wandb_logger.experiment.config.update(cfg)

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
                    logger=[wandb_logger, csv_logger],
                    callbacks=[ early_stop, checkpoint_callback, DeviceStatsMonitor() ]  # 
                    )

classifier = models.SentimentClassifier(tokenizer=tokenizer, 
                                        hyperparams=cfg.hyperparameters)

# log gradients and model topology
wandb_logger.watch(classifier)

trainer.fit(classifier, train_dataloader, val_dataloader)

wandb_logger.experiment.unwatch(classifier)

if cfg.logging.wandb:
    # [optional] finish the wandb run, necessary in notebooks
    wandb.finish()