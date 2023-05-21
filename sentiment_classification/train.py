

def main():
    import os
    import torch
    import dataflow
    import model as models
    import utils
    import custom_callbacks

    import lightning as L
    from omegaconf import OmegaConf
    import time
    import wandb
    import pickle
    import argparse

    import psutil
    import logging


    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", type=str, help="Config path for training")
    parser.add_argument("--server_log_file", type=str, help="Log file for instance output")
    args = parser.parse_args()

    run_logger = utils.create_logger(__name__)


    cfg_path = args.cfg_path  # 'configs/local-canine-backbone.yaml'
    cfg = OmegaConf.load(cfg_path)

    server_log_file = args.server_log_file
    cfg.datafiles.server_log_file = server_log_file

    # Function for setting the seed
    L.seed_everything(cfg.seed)


    # -----------------------------------------------
    # MACHINE PRECISION

    def set_torch_precision():
        '''
        This is recommended by Lightning
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
                float32_matmul_precision = 'medium'
                run_logger.info(f"This device has Tensor Cores! Setting precision to {float32_matmul_precision}...")
                torch.set_float32_matmul_precision('medium')
            else:
                run_logger.info("This device does not have Tensor Cores :(")

    set_torch_precision()

    def set_trainer_precision():

        if torch.cuda.is_available() == False and str(cfg.hyperparameters.trainer.precision) != '32':
            run_logger.warning("CUDA not available, Lightning Trainer precision being set to standard 32-bit...")
            cfg.hyperparameters.trainer.precision = '32'
        else:
            run_logger.info(f"Lightning Trainer Precision being set to {cfg.hyperparameters.trainer.precision}")

    set_trainer_precision()

    # -----------------------------------------------------------------------
    # DATA

    def set_num_workers():
        # Get the number of CPU cores
        num_cpus = os.cpu_count()

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
        # num_workers = set_num_workers()
        num_workers = 2
        cfg.hyperparameters.num_workers = num_workers
        
    run_logger.info(f"num_workers assigned for DataLoader: {cfg.hyperparameters.num_workers}")


    dm = dataflow.TweetDataModule(cfg)

    dm.prepare_data()

    cfg.data_processing.label_decode_map = dm.label_decode_map
    cfg.data_processing.tokenizer.kwargs.vocab_size = dm.tokenizer.vocab_size
    cfg.hyperparameters.backbone.kwargs.input_size = dm.tokenizer.vocab_size


    # ---------------------------------------------------------------------
    # LOGGERS

    if 'loggers' in cfg:
        loggers = {}
        if 'wandb_logger' in cfg.loggers:
            wandb_logging = True
            if cfg.loggers.wandb_logger.kwargs.offline == False:
                wandb.login(key=os.environ['WANDB_API_KEY'])

        if 'csv_logger' in cfg.loggers:
            cfg.loggers.csv_logger.kwargs.version = time.strftime("%Y-%m-%d__%H-%M")


        for logger, values in cfg.loggers.items():
            if logger not in loggers:
                loggers[logger] = (utils.load_obj(values.object)(**values.get('kwargs', {})))

        if wandb_logging:
            loggers['wandb_logger'].experiment.config.update(cfg)


    # ---------------------------------------------------------------------
    # CALLBACKS

    if 'callbacks' in cfg:
        callbacks = {}
        if 'slack_callback' in cfg.callbacks:
            if cfg.callbacks.slack_callback:
                callbacks['slack_callback'] = custom_callbacks.SlackCallback(cfg=OmegaConf.to_yaml(cfg),
                                                                server_log_file=server_log_file
                                                                )
        if 'print_table_metrics_callback' in cfg.callbacks:
            if cfg.callbacks.print_table_metrics_callback:
                callbacks['print_table_metrics_callback'] = custom_callbacks.PrintTableMetricsCallback()

        for callback, values in cfg.callbacks.items():
            if callback not in callbacks:
                callbacks[callback] = (utils.load_obj(values.object)(**values.get('kwargs', {})))



    # -----------------------------------------------------------------------
    # # PROFILER
    # from lightning.pytorch.profilers import PyTorchProfiler

    # pytorch_profiler = PyTorchProfiler(profile_memory = True, 
    #                                    sort_by_key='cuda_memory_usage',
    #                                    )
    # cfg.hyperparameters.trainer.profiler = pytorch_profiler


    # -----------------------------------------------------
    # TRAIN

    trainer = L.Trainer(logger=list(loggers.values()),
                        callbacks=list(callbacks.values()),
                        **cfg.hyperparameters.get('trainer', {}),
                        )
    
    # # Save tokenizer as a pickle file to the log directory
    # tokenizer_save_path = os.path.join(trainer.log_dir, trainer.logger.name, trainer.logger.version)
    # os.makedirs(tokenizer_save_path, exist_ok=True)
    # with open(os.path.join(tokenizer_save_path, 'tokenizer.pkl'), 'wb') as f:
    #     pickle.dump(dm.tokenizer, file=f)

    # cfg.datafiles.tokenizer_save_path = tokenizer_save_path
    # run_logger.info(f"Tokenizer saved to {tokenizer_save_path}")
    

    # Initialise model
    classifier = models.SentimentClassifier(tokenizer=dm.tokenizer, 
                                            hyperparams=cfg.hyperparameters)

    # log gradients and model topology
    if wandb_logging:
        loggers['wandb_logger'].watch(classifier)

        cfg.datafiles.saved_ckpt_dir = os.path.join(loggers['wandb_logger'].name, loggers['wandb_logger'].version, 'checkpoints')
        run_logger.info(f"Model ckpts saved to {cfg.datafiles.saved_ckpt_dir}")

    # # Save config file to log directory
    # config_save_path = os.path.join(trainer.log_dir, trainer.logger.name, trainer.logger.version, 'config.yaml')
    # with open(config_save_path, 'w') as f:
    #     OmegaConf.save(config=cfg, f=f)
        
    # run_logger.info(f"Config saved to {config_save_path}")

    trainer.fit(classifier, datamodule=dm)

    if wandb_logging:
        loggers['wandb_logger'].experiment.unwatch(classifier)

    # # ----------------------------------------------------
    # # TEST

    trainer.test(classifier, datamodule=dm)

    if 'wandb_logger' in loggers:
        # [optional] finish the wandb run, necessary in notebooks
        wandb.finish()


if __name__ == '__main__':
    main()
