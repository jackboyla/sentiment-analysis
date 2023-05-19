
def main():
    import os
    import torch
    import dataflow as dataflow
    import model as models
    import utils

    import lightning as L
    from omegaconf import OmegaConf

    import argparse



    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", type=str, help="Config path for training")
    parser.add_argument("--server_log_file", type=str, help="Log file for instance output")
    args = parser.parse_args()

    logger = utils.create_logger(__name__)


    cfg_path = args.cfg_path  # 'configs/local-canine-backbone.yaml'
    cfg = OmegaConf.load(cfg_path)

    logger.info(f"Loading config from {cfg_path}")

if __name__ == '__main__':
    main()