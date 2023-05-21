
def main():
    import os
    import torch
    import dataflow as dataflow
    import model as models
    import utils
    import glob
    import pickle

    import lightning as L
    from omegaconf import OmegaConf

    import argparse

    logger = utils.create_logger(__name__)

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg-path", type=str, help="Config path for training")
    parser.add_argument("--input", type=str, help="File to make predictions on")
    parser.add_argument("--output", type=str, help="File to write predictions to")
    args = parser.parse_args()

    cfg_path = args.cfg_path
    cfg = OmegaConf.load(cfg_path)

    logger.info(f"Loaded config from {cfg_path}")

    # Load Tokenizer
    with open(cfg.datafiles.tokenizer_file, 'rb') as f:
        tokenizer = pickle.load(f)

    # Get latest checkpoint
    model_ckpts = glob.glob(f'{cfg.datafiles.saved_ckpt_dir}/*') 
    latest_ckpt = max(model_ckpts, key=os.path.getctime)
    logger.info(f"Found latest checkpoint at {latest_ckpt}")
    
    # Load model from checkpoint
    model = models.SentimentClassifier.load_from_checkpoint(latest_ckpt, 
                                                            tokenizer=tokenizer, 
                                                            hyperparams=cfg.hyperparameters)
    model.eval()

    outputs = []
    with open(args.input, 'r') as f:
        with torch.no_grad():
            # Iterate through each line in the input file
            for line in f:

                # Pass the input through the model
                out = model(line)
        
                # Convert the outputs to a readable format if needed
                out = cfg.data_processing.decode_label_map[out]
                outputs.append(out)
        

    with open(args.output, 'w') as f:
        # Write the output to the output file
        f.write([out + '\n' for out in outputs])

    logger.info(f"Saved predictions to {args.output}")


if __name__ == '__main__':
    main()