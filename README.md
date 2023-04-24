# Sentiment Analysis at the Character Level

> Character level models give up the semantic information that words have, as well as the plug and play ecosystem of pre-trained word vectors. In exchange, character level deep learning models provide two fundamental advantages. They alleviate vocabulary problems we encounter on the input of our model, and they remove the computational bottleneck at the output of our model. [source](https://www.lighttag.io/blog/character-level-NLP/)

> A language model aims to predict the next token given the previous tokens. As is standard, the final layer computes a softmax over every token in the vocabulary. With a large vocabulary, the softmax step and associated gradient calculations become the performance bottleneck in training a language model.

Character-level models solve this by having a smaller vocabulary.

> Character level models are not a panacea and come with their own set of drawbacks. The two most glaring ones are the lack of semantic content of the input (characters are meaningless) and the growth in the length of our inputs. The average English word has five characters meaning that dependent on architecture, we can expect a 5X increase in compute requirements.

The Reformer architecture is designed to be memory-efficient and scalable, making it a good choice for large-scale language modeling tasks