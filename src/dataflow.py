import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import re


# https://huggingface.co/google/reformer-enwik8
'''
The model is a language model that operates on characters. 
Therefore, this model does not need a tokenizer. The following function can instead be used for encoding and decoding:
'''

class CharacterTokenizer():
    def __init__(self, pad_token_id=0, encoding="ISO-8859-1"):
        self.pad_token_id = torch.tensor(pad_token_id)
        self.encoding = encoding

    # Encoding
    def encode(self, list_of_strings):
        '''
        Unlike normal HF tokenizers, which return:
          dict_keys(['input_ids', 'token_type_ids', 'attention_mask'])
        This returns only (input_ids, attention_mask)
        '''
        max_length = max([len(string) for string in list_of_strings])

        # create emtpy tensors
        attention_masks = torch.zeros((len(list_of_strings), max_length), dtype=torch.long)
        input_ids = torch.full((len(list_of_strings), max_length), self.pad_token_id, dtype=torch.long)

        for idx, string in enumerate(list_of_strings):
            # make sure string is in byte format
            orig_string = string

            if not isinstance(string, bytes):
                string = str.encode(string, encoding=self.encoding)

            try:
                input_ids[idx, :len(string)] = torch.tensor([x + 2 for x in string])
            except Exception as error:
                print('Caught this error: ' + repr(error))
                print(f"\nWas working on this: {orig_string}, as {string} {idx}")
                print(f"List of strings: \n{list_of_strings}")

            attention_masks[idx, :len(string)] = 1

        return input_ids.squeeze(), attention_masks.squeeze()
        
    # Decoding
    def decode(self, outputs_ids):
        decoded_outputs = []
        for output_ids in outputs_ids.tolist():
            # transform id back to char IDs < 2 are simply transformed to ""
            decoded_outputs.append("".join([chr(x - 2) if x > 1 else "" for x in output_ids]))
        return decoded_outputs

def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def remove_mentions_hashtags(text):
    mention_pattern = re.compile(r'@\w+')
    hashtag_pattern = re.compile(r'#\w+')
    text = mention_pattern.sub(r'', text)
    text = hashtag_pattern.sub(r'', text)
    return text

class TweetDataset(Dataset):
    def __init__(self, tweets, labels, tokenizer):
        tweets = tweets.apply(remove_urls)
        self.tweets = tweets.apply(remove_mentions_hashtags)
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, idx):
        tweets = self.tweets[idx]
        labels = self.labels[idx]

        encoded_input_ids, attention_masks = self.tokenizer.encode(tweets)
        return {
            'tweet': tweets,
            'input_ids': encoded_input_ids,
            'attention_mask': attention_masks,
            'labels': labels
                }
    
