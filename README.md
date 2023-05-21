# Sentiment Analysis at the Character Level

## Introduction & Motivation
Character level models give up the semantic information that words have, as well as the plug and play ecosystem of pre-trained word emeddings. 

In exchange, character level NLP models solve a number of problems:

1. They alleviate vocabulary problems we encounter on the input of our model, and for Sequence-to-Sequence models they remove the computational bottleneck at the output layer of our model by having a much smaller vocabulary.

2. They can be easily applied to languages such as Thai, Lao, Chinese, and Khmer, which do not follow a structure like English. Word- and sub-word- models must have specialised tokenizers crafted for each language, whereas a character-level model operates as normal.


> Character level models are not a panacea and come with their own set of drawbacks. The two most glaring ones are the lack of semantic content of the input (characters are meaningless) and the growth in the length of our inputs. The average English word has five characters meaning that dependent on architecture, we can expect a 5X increase in compute requirements. [source](https://www.lighttag.io/blog/character-level-NLP/)

## Getting Started
```
# create a new environment
conda create -n sentiment-analysis python=3.11
conda activate sentiment-analysis

# Installation 
pip install -r requirements.txt
```

## Usage
To start training, you can set up a YAML config file, either using one from `configs/` or create your own.

*Note*: I have implemented the code to provide maximum flexibility with config files. Objects such as backbone models, optimizers, and tokenizers are read from the config file, like so:
```
# Specified in YAML

optimizer:
    object: torch.optim.AdamW

# Retrieved during model build
optimizer = utils.load_obj(self.cfg.optimizer.object)
```

To run training locally:
```
# [optional]
export SLACK_HOOK=...
export WANDB_API_KEY=...

# run training
python src/train.py --cfg_path configs/intro-transformer-backbone.yaml --server_log_file transformer_training_run.log
```

Alternatively, you can train on a Vast.ai instance. Read [VastAIRun.md](VstAIRun.md) for more info.


## Report

Each model is evaluated on a test set which consists of 20% of the total dataset.

The loss (Categorical Cross Entropy), Accuracy and F1 Score are recorded.

### Results


### Shortcomings (and possible solutions)

The 

#### LLMs at the Character Level
The [CANINE Model](https://arxiv.org/abs/2103.06874) is a unqiue pretrained model in that it does not require a rigid tokenizer. Pretrained on 104 languages using a masked language modeling (MLM) objective, it processes language at a character level: each character is turned into its Unicode code point. See the linked paper and the [CANINE HuggingFace page](https://huggingface.co/google/canine-c) for further details.