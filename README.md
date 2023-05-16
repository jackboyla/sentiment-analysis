# Sentiment Analysis at the Character Level

## Intro
Character level models give up the semantic information that words have, as well as the plug and play ecosystem of pre-trained word vectors. In exchange, character level deep learning models provide two fundamental advantages. They alleviate vocabulary problems we encounter on the input of our model, and they remove the computational bottleneck at the output of our model by having a much smaller vocabulary. [source](https://www.lighttag.io/blog/character-level-NLP/)


> Character level models are not a panacea and come with their own set of drawbacks. The two most glaring ones are the lack of semantic content of the input (characters are meaningless) and the growth in the length of our inputs. The average English word has five characters meaning that dependent on architecture, we can expect a 5X increase in compute requirements.

## LLMs at the Character Level
The [CANINE Model](https://arxiv.org/abs/2103.06874) is a unqiue pretrained model in that it does not require a tokenizer. Pretrained, on 104 languages using a masked language modeling (MLM) objective, it processes language at a character level: each character is turned into its Unicode code point. See the linked paper and the [CANINE HuggingFace page](https://huggingface.co/google/canine-c) for further details.

## Setup
```
# create a new environment
conda create -n sentiment-analysis python=3.11
conda activate sentiment-analysis

# Installation 
pip install -r requirements.txt
```

## Usage
To start training, you can set up a YAML config file, either using one from `configs/` or create your own.

To run locally:
```
# [optional]
export SLACK_HOOK=...
export WANDB_API_KEY=...

# run training
python src/train.py --cfg_path configs/dev.yaml --server_log_file training_log.log
```

Alternatively, you can train on a Vast.ai instance:
1. Inside your Vast.ai console, choose an image  
2. Add environment variables to the image like so:
```-e SLACK_HOOK=... -e WANDB_API_KEY=...```

2. Copy the contents of `vast-ai-run.sh` into the startup script.
3. Rent an instance and training will commence automatically. 

Slack and WandB will provide updates if the relevant details are supplied. The instance will complete train.py with all specified config files. After each training run finishes, the files are uploaded via transfer.sh, where they will be available for download. The instance will then destroy itself to avoid wasting resources.

The transfer.sh link will be provided via Slack.