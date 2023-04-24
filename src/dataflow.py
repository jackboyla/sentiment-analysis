
from torch.utils.data import Dataset
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
    
