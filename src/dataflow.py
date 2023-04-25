
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import re

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
        self.tweets = self.tweets
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, idx):
        tweets = self.tweets[idx]
        labels = self.labels[idx]

        # encoded_input_ids, attention_masks = self.tokenizer.encode(tweets)
        encoded_input = self.tokenizer(tweets)
        encoded_input.update({'tweet': tweets, 'labels': labels})
        return encoded_input
    

def collate_fn(batch, input_pad_token_id=0):
    '''
    dynamically padding input sequences per batch
    batch is alist of dicts
    '''

    attention_masks = [torch.tensor(sample['attention_mask']) for sample in batch]
    input_ids = [torch.tensor(sample['input_ids']) for sample in batch]
    labels = [sample['labels'] for sample in batch]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=input_pad_token_id)
    attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0.0)
        

    return {'input_ids': input_ids, 'attention_mask': attention_masks}, torch.tensor(labels)