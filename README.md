# Sentiment Analysis at the Character Level

[GITHUB REPO LINK](https://github.com/jackboyla/sentiment-analysis)

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
cd <folder you have cloned/downloaded repo to> && make dev
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

### Train

To run training locally:
```
# [optional]
export SLACK_HOOK=...
export WANDB_API_KEY=...

# run training
python sentiment_classification/train.py --cfg_path configs/intro-transformer-backbone.yaml --server_log_file transformer_training_run.log
```

Alternatively, you can train on a Vast.ai instance. Read [VastAIRun.md](VstAIRun.md) for more info.

### Inference

To run inference, you must specify a config file to load the model from:

```
python sentiment_classification/predict.py --cfg-path=wandb-sentiment-analysis/4q1luinh/config.yaml --input=input.txt --output=preds.txt
```

This will produce a file `output.txt` that reads:

```
positive
negative
...
```


## Report

I have tried to set up a project via the [Aylien Data Science Quickstarter tool](https://github.com/AYLIEN/datascience-project-quickstarter) (which I love!), but unfortunately I'm running into too many dependency issues on Windows to get everything working. 

I also caught the flu this week, which meant I couldn't work on this project as much as I'd like :( But I hope to get back to it soon.

I have trained a Transformer (`transformer.py`), an LSTM (`lstm.py`), and a fine-tuned CANINE model (loaded in `model.py`) on the provided data. 

I have also trained a simple embedding + Linear model (`simple_embedding.py`) to debug a major issue of vanishing gradients (outlined below).

All model training and evaluation is carried out using PyTorch Lightning.


### Results

Each model is evaluated on a test set which consists of 20% of the total dataset.

The loss, Accuracy and F1 Score are logged according to config specifications (default is WandB logging).

Right now, there are no meangingful evaluation results, as the models have struggled to learn anything.


### Shortcomings (and possible solutions)

There is a plateauing effect with the loss as it is not converging to ~0, but some arbitrary value. 

I have tried using a Random Dataset which should be extremely easy to learn, and the problem remains.

I have checked the gradients during training and every model architecture suffers from a vanishing gradient issue, despite changes to the loss function, model capacity, LRs, schedulers, initialisations.

I have tried overfitting the model on the first batch to the same result. I have also tried balancing my data using torch's WeightedRandomSampler.



#### LLMs at the Character Level

I was surprised to find that there aren't many pretrained character-level models available online.

The [CANINE Model](https://arxiv.org/abs/2103.06874) is a unqiue pretrained model in that it does not require a rigid tokenizer. Pretrained on 104 languages using a masked language modeling (MLM) objective, it processes language at a character level: each character is turned into its Unicode code point. See the linked paper and the [CANINE HuggingFace page](https://huggingface.co/google/canine-c) for further details.