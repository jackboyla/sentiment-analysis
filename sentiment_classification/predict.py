
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
    parser.add_argument("--cfg-path", type=str, help="Config path of model you want to use for inference")
    parser.add_argument("--input", type=str, help="File to make predictions on")
    parser.add_argument("--output", type=str, help="File to write predictions to")
    args = parser.parse_args()

    cfg_path = args.cfg_path
    cfg = OmegaConf.load(cfg_path)

    logger.info(f"Loaded config from {cfg_path}")

    # Get latest checkpoint
    model_ckpts = glob.glob(f'{cfg.datafiles.saved_ckpt_dir}/*') 
    latest_ckpt = max(model_ckpts, key=os.path.getctime)
    logger.info(f"Found latest checkpoint at {latest_ckpt}")
    
    # Load model from architecture and state_dict
    with open(cfg.datafiles.architecture_save_path, 'rb') as f:
        model = pickle.load(f)

    latest_ckpt = torch.load(latest_ckpt)
    model.load_state_dict(latest_ckpt['state_dict'])
    model.eval()

    # Begin Inference
    outputs = []
    label_decode_map = {v: k for k, v in cfg.data_processing.label_decode_map.items()}
    with open(args.input, 'r', encoding="utf8") as f:

        with torch.no_grad():
            # Iterate through each line in the input file
            for line in f:

                # Pass the input through the model
                out = model(line)[0]
                # print(f"Text: {line}Prediction: {out}\n")
        
                # Convert the outputs to a readable format
                out = label_decode_map[out]
                outputs.append(out)
        

    # Write Predictions to output file
    with open(args.output, 'w') as f:
        for prediction in outputs:
            f.write(f"{prediction}\n")

    logger.info(f"Saved predictions to {args.output}")


if __name__ == '__main__':
    main()