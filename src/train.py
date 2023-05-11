

def main():
    import os
    import torch
    import dataflow as dataflow
    import model as models
    import utils

    import lightning as L
    from omegaconf import OmegaConf
    from lightning.pytorch.callbacks import DeviceStatsMonitor
    import time
    import wandb
    import argparse

    import multiprocessing
    import psutil


    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", type=str, help="Config path for training")
    parser.add_argument("--server_log_file", type=str, help="Log file for instance output")
    args = parser.parse_args()


    cfg_path = args.cfg_path  # 'configs/local-canine-backbone.yaml'
    cfg = OmegaConf.load(cfg_path)

    server_log_file = args.server_log_file

    def set_torch_precision():
        '''
        This is recommended by Lightning, but causes OOM crashes when running on Vast.ai instances
        '''
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
                torch.set_float32_matmul_precision('medium')
            else:
                print("This device does not have Tensor Cores :(")

    # set_torch_precision()


    def set_num_workers():
        # Get the number of CPU cores
        num_cpus = multiprocessing.cpu_count()

        # Check the available memory
        available_memory_gb = psutil.virtual_memory().available / 1024 / 1024 / 1024

        # Determine the number of workers based on the available memory
        # In practice a factor of 4 seems to work relatively well for Dataloader num_workers
        if available_memory_gb < 4:
            num_workers = 0
        elif available_memory_gb < 8:
            num_workers = min(num_cpus, 4)
        else:
            num_workers = min(num_cpus, 8)

        # # Set the number of workers for PyTorch data loaders
        # torch.set_num_threads(num_workers)

        available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
        num_workers *= len(available_gpus)
        
        return num_workers

    if 'num_workers' not in cfg.hyperparameters:
        num_workers = set_num_workers()
        cfg.hyperparameters.num_workers = num_workers
    print(f"num_workers assigned for DataLoader: {cfg.hyperparameters.num_workers}")


    # -------------------------------------------------
    # DATA

    dm = dataflow.TweetDataModule(cfg)

    # --------------------------------------------------
    # LOGGERS AND CALLBACKS

    if 'wandb_logger' in cfg.loggers:
        wandb_logging = True
        if cfg.loggers.wandb_logger.kwargs.offline == False:
            wandb.login(key=os.environ['WANDB_API_KEY'])

    if 'csv_logger' in cfg.loggers:
        cfg.loggers.csv_logger.kwargs.version = time.strftime("%Y-%m-%d__%H-%M")

    loggers = {}
    for logger, values in cfg.loggers.items():
        loggers[logger] = (utils.load_obj(values.object)(**values.kwargs))


    if 'callbacks' in cfg:
        callbacks = {}
        if 'slack_callback' in cfg.callbacks:
            callbacks['slack_callback'] = utils.SlackCallback(webhook_url=os.environ['SLACK_HOOK'], 
                                            cfg=OmegaConf.to_yaml(cfg),
                                            server_log_file=server_log_file
                                            )
        if 'print_table_metrics_callback' in cfg.callbacks:
            callbacks['print_table_metrics_callback'] = utils.PrintTableMetricsCallback()
        if 'device_stats_monitor_callback' in cfg.callbacks:
            callbacks['device_stats_monitor_callback'] = DeviceStatsMonitor()

        for callback, values in cfg.callbacks.items():
            if callback not in callbacks:
                callbacks[callback] = (utils.load_obj(values.object)(**values.kwargs))


    from lightning.pytorch.profilers import PyTorchProfiler

    pytorch_profiler = PyTorchProfiler(
                                       dirpath='data/', 
                                       filename='pytorch_profiler', 
                                       profile_memory = True, 
                                       sort_by_key='cuda_memory_usage',
                                    #    with_stack = True
                                       )


    # ----------------------------
    # TRAIN

    trainer = L.Trainer(logger=list(loggers.values()),
                        # callbacks=list(callbacks.values()),
                        profiler=pytorch_profiler,
                        **cfg.hyperparameters.trainer
                        )

    classifier = models.SentimentClassifier(tokenizer=dm.tokenizer, 
                                            hyperparams=cfg.hyperparameters)

    # # log gradients and model topology
    # if wandb_logging:
    #     loggers['wandb_logger'].watch(classifier)

    trainer.fit(classifier, datamodule=dm)

    # if wandb_logging:
    #     loggers['wandb_logger'].experiment.unwatch(classifier)

    # # Save config file to CSV log directory
    # with open(os.path.join(*[cfg.loggers.csv_logger.kwargs.save_dir, 
    #                         cfg.loggers.csv_logger.kwargs.name, 
    #                         cfg.loggers.csv_logger.kwargs.version, 
    #                         'config.yaml']), 'w') as f:
    #     OmegaConf.save(config=cfg, f=f)

    trainer.test(classifier, datamodule=dm)

    if 'wandb_logger' in loggers:
        # [optional] finish the wandb run, necessary in notebooks
        wandb.finish()


if __name__ == '__main__':
    main()
