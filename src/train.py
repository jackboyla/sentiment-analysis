
import os
import multiprocessing

os.environ['CPU_COUNT'] = str(multiprocessing.cpu_count())

def main():
    import importlib
    import sys
    import os
    import torch
    from torch.utils.data import DataLoader
    import dataflow as dataflow
    import model as models
    import utils

    import lightning as L

    from sklearn.model_selection import train_test_split

    import pandas as pd
    from omegaconf import OmegaConf
    from transformers import CanineTokenizer

    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, ModelSummary, DeviceStatsMonitor
    from lightning.pytorch.loggers import WandbLogger, CSVLogger
    import time
    import wandb
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", type=str, help="Config path for training")
    parser.add_argument("--server_log_file", type=str, help="Log file for instance output")
    args = parser.parse_args()

    if torch.cuda.is_available():
        # Specify the device you want to check
        device = torch.device("cuda:0")  # Change 0 to the index of your CUDA device if you have multiple GPUs

        # Get the device properties
        device_props = torch.cuda.get_device_properties(device)

        ''' 
            Check if the device has Tensor Cores
            According to NVIDIA's documentation, 
            Tensor Cores were introduced in the Volta architecture (CUDA Compute Capability 7.0 and above), 
            so we check if the device's major version is 7 or greater and the minor version is 0 or greater.
        '''
        if device_props.major >= 7 and device_props.minor >= 0:
            print("This device has Tensor Cores!\n Setting precision to medium...")
            torch.set_float32_matmul_precision('medium' | 'high')
        else:
            print("This device does not have Tensor Cores :(")


    cfg_path = args.cfg_path  # 'configs/local-canine-backbone.yaml'
    cfg = OmegaConf.load(cfg_path)

    server_log_file = args.server_log_file


    dm = dataflow.TweetDataModule(cfg)

    # DATASET_COLUMNS = ["target", "ids", "date", "flag", "user", "text"]
    # df = pd.read_csv(
    #     cfg.datafiles.data_dir,
    #     encoding=cfg.datafiles.dataset_encoding,
    #     names=DATASET_COLUMNS
    #     )

    # decode_map = {0: 0, 4: 1}
    # def binarize_sentiment(label):
    #     return decode_map[int(label)]

    # df['target'] = df.target.apply(lambda x: binarize_sentiment(x))

    # train_size = int(cfg.data_processing.train_size if cfg.data_processing.train_size > 1.0 else cfg.data_processing.train_size*len(df))
    # val_size = int(cfg.data_processing.val_size if cfg.data_processing.val_size > 1.0 else cfg.data_processing.val_size*len(df))
    # test_size = int(cfg.data_processing.test_size if cfg.data_processing.test_size > 1.0 else cfg.data_processing.test_size*len(df))

    # train_df, test_df = train_test_split(df, train_size=train_size+val_size, test_size=test_size, random_state=42, stratify=df['target'])
    # train_df, val_df = train_test_split(train_df, train_size=train_size, test_size=val_size, random_state=42, stratify=train_df['target'])

    # train_df.reset_index(inplace=True, drop=True)
    # val_df.reset_index(inplace=True, drop=True)
    # test_df.reset_index(inplace=True, drop=True)



    # # single sequence: [CLS] X [SEP]
    # tokenizer = CanineTokenizer.from_pretrained("google/canine-c", 
    #                                             return_tensors='pt')

    # # Build Datasets
    # train_dataset = dataflow.TweetDataset(train_df['text'], train_df['target'], tokenizer)
    # val_dataset = dataflow.TweetDataset(val_df['text'], val_df['target'], tokenizer)
    # test_dataset = dataflow.TweetDataset(test_df['text'], test_df['target'], tokenizer)


    # # Build DataLoaders
    # train_dataloader = DataLoader(train_dataset, 
    #                               batch_size=cfg.hyperparameters.batch_size, 
    #                               shuffle=True, 
    #                               collate_fn=lambda b: dataflow.collate_fn(b, input_pad_token_id=tokenizer.pad_token_id)
    #                               )
    # val_dataloader = DataLoader(val_dataset, batch_size=cfg.hyperparameters.batch_size, shuffle=False, collate_fn=lambda b: dataflow.collate_fn(b, input_pad_token_id=tokenizer.pad_token_id))
    # test_dataloader = DataLoader(test_dataset, batch_size=cfg.hyperparameters.batch_size, shuffle=False, collate_fn=lambda b: dataflow.collate_fn(b, input_pad_token_id=tokenizer.pad_token_id))


    # if cfg.logging.wandb:
    #     # # DO NOT login if you want to log offline
    #     # wandb.login(key=os.environ['WANDB_API_KEY'])
    #     wandb_logger = WandbLogger(project=cfg.logging.project_name, 
    #                                offline=True
    #                                )
        
    # slack_callback = utils.SlackCallback(webhook_url=os.environ['SLACK_HOOK'], 
    #                                      cfg=OmegaConf.to_yaml(cfg),
    #                                      server_log_file=server_log_file
    #                                      )

    # cfg.loggers.csv_logger.kwargs.log_name = 'lightning_logs'
    cfg.loggers.csv_logger.kwargs.version = time.strftime("%Y-%m-%d__%H-%M")

    loggers = {}
    for logger, values in cfg.loggers.items():
        loggers[logger] = (utils.load_obj(values.object)(**values.kwargs))
    # csv_logger = CSVLogger(cfg.logging.log_dir, name=log_name, version=version)


    # print(f"Training on {len(dm.train_df)} examples...")


    # early_stop = EarlyStopping(cfg.callbacks.early_stopping.monitor, 
    #                            patience=cfg.callbacks.early_stopping.patience, 
    #                            verbose=True, 
    #                            min_delta=cfg.callbacks.early_stopping.min_delta)

    # checkpoint_callback = ModelCheckpoint(save_top_k=cfg.callbacks.checkpoint.save_top_k, 
    #                                       monitor=cfg.callbacks.checkpoint.monitor)


    callbacks = {}
    if cfg.callbacks.slack_callback:
        callbacks['slack_callback'] = utils.SlackCallback(webhook_url=os.environ['SLACK_HOOK'], 
                                        cfg=OmegaConf.to_yaml(cfg),
                                        server_log_file=server_log_file
                                        )
    if cfg.callbacks.print_table_metrics_callback:
        callbacks['print_table_metrics_callback'] = utils.PrintTableMetricsCallback()
    if cfg.callbacks.device_stats_monitor_callback:
        callbacks['device_stats_monitor_callback'] = DeviceStatsMonitor()

    for callback, values in cfg.callbacks.items():
        if callback not in callbacks:
            callbacks[callback] = (utils.load_obj(values.object)(**values.kwargs))



    trainer = L.Trainer(logger=list(loggers.values()),
                        callbacks=list(callbacks.values()), 
                        **cfg.hyperparameters.trainer
                        )

    classifier = models.SentimentClassifier(tokenizer=dm.tokenizer, 
                                            hyperparams=cfg.hyperparameters)

    # log gradients and model topology
    if 'wandb_logger' in loggers:
        loggers['wandb_logger'].watch(classifier)

    trainer.fit(classifier, datamodule=dm)

    if 'wandb_logger' in loggers:
        loggers['wandb_logger'].experiment.unwatch(classifier)

    # Save config file to CSV log directory
    with open(os.path.join(*[cfg.loggers.csv_logger.kwargs.save_dir, 
                            cfg.loggers.csv_logger.kwargs.name, 
                            cfg.loggers.csv_logger.kwargs.version, 
                            'config.yaml']), 'w') as f:
        OmegaConf.save(config=cfg, f=f)

    trainer.test(classifier, datamodule=dm)

    if 'wandb_logger' in loggers:
        # [optional] finish the wandb run, necessary in notebooks
        wandb.finish()


if __name__ == '__main__':
    main()