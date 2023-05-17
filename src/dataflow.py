
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import re
import utils
import lightning as L
from sklearn.model_selection import train_test_split

import pandas as pd
from transformers import AutoTokenizer
from functools import partial
import html

def convert_html_entities(text):
    return html.unescape(text)

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
        self.tweets = tweets
        self.tweets = self.tweets.apply(convert_html_entities)
        self.tweets = self.tweets.apply(remove_urls)
        self.tweets = self.tweets.apply(remove_mentions_hashtags)
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, idx):
        tweets = self.tweets[idx]
        labels = self.labels[idx]

        encoded_input = self.tokenizer(tweets)
        encoded_input.update({'tweet': tweets, 'labels': labels})
        return encoded_input
    
def embed_bag_collate_fn(batch, input_pad_token_id=0):
    '''
    Colatte fn for prepping batch before EmbeddingBag Layer
    '''

    attention_masks = [torch.tensor(sample['attention_mask']) for sample in batch]
    attention_masks = torch.cat(attention_masks, dim=0)
    input_ids = [torch.tensor(sample['input_ids']) for sample in batch]
    offsets = torch.tensor([0] + [len(ids) for ids in input_ids[:-1]])
    offsets = torch.cumsum(offsets, dim=0)
    input_ids = torch.cat(input_ids, dim=0)
    labels = [sample['labels'] for sample in batch]


    return {'input_ids': input_ids, 'offsets': offsets, 'attention_mask': attention_masks}, torch.tensor(labels)

def partial_embed_bag_collate_fn(b, input_pad_token_id):
    return embed_bag_collate_fn(b, input_pad_token_id=input_pad_token_id)


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

def partial_collate_fn(b, input_pad_token_id):
    return collate_fn(b, input_pad_token_id=input_pad_token_id)



class TweetDataModule(L.pytorch.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_workers = self.cfg.hyperparameters.num_workers
        self.pin_memory = self.cfg.hyperparameters.pin_memory

        self.label_decode_map = {'negative': 0, 
                                 'neutral': 1,
                                 'positive': 2}


        if 'transformers' in cfg.data_processing.tokenizer.object:
            # single sequence: [CLS] sequence_ids [SEP]
            self.tokenizer = AutoTokenizer.from_pretrained(**self.cfg.data_processing.tokenizer.kwargs)
        else:
            self.tokenizer = utils.load_obj(self.cfg.data_processing.tokenizer.object)
        


    def prepare_data(self):
        """
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html#prepare-data
        """

        df = pd.DataFrame()

        for file, label in self.cfg.datafiles.data_dirs.items():
            temp_df = pd.read_csv(file, sep='\t', **self.cfg.datafiles.get('kwargs', {})
                                )
            if label in ['positive', 'neutral', 'negative']:
                temp_df['label'] = label
            df = pd.concat([df, temp_df])

        df.reset_index(inplace=True, drop=True)

        def encode_sentiment(label):
            return self.label_decode_map[label]

        df['target'] = df['label'].apply(lambda x: encode_sentiment(x))

        train_size = int(self.cfg.data_processing.train_size if self.cfg.data_processing.train_size > 1.0 else self.cfg.data_processing.train_size*len(df))
        val_size = int(self.cfg.data_processing.val_size if self.cfg.data_processing.val_size > 1.0 else self.cfg.data_processing.val_size*len(df))
        test_size = int(self.cfg.data_processing.test_size if self.cfg.data_processing.test_size > 1.0 else self.cfg.data_processing.test_size*len(df))
        print(f"Train/Val/Test Splits: {train_size}/{val_size}/{test_size}")

        self.train_df, self.test_df = train_test_split(df, train_size=train_size+val_size, test_size=test_size, random_state=42, stratify=df['target'])
        self.train_df, self.val_df = train_test_split(self.train_df, train_size=train_size, test_size=val_size, random_state=42, stratify=self.train_df['target'])

        self.train_df.reset_index(inplace=True, drop=True)
        self.val_df.reset_index(inplace=True, drop=True)
        self.test_df.reset_index(inplace=True, drop=True)


    def setup(self, stage: str):
        """
        Method to setup your datasets, here you can use whatever dataset class you have defined in Pytorch and prepare the data in order to pass it to the loaders later
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html#setup
        """
        if stage == "fit":
            self.train_dataset = TweetDataset(self.train_df['text'], self.train_df['target'], self.tokenizer)
            self.val_dataset = TweetDataset(self.val_df['text'], self.val_df['target'], self.tokenizer)
            
        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.test_dataset = TweetDataset(self.test_df['text'], self.test_df['target'], self.tokenizer)


    def train_dataloader(self):
        return DataLoader(self.train_dataset, 
                              batch_size=self.cfg.hyperparameters.batch_size, 
                              shuffle=True, 
                            #   collate_fn=lambda b: collate_fn(b, input_pad_token_id=self.tokenizer.pad_token_id),
                              collate_fn=partial(partial_collate_fn, input_pad_token_id=self.tokenizer.pad_token_id),
                              num_workers=self.num_workers,
                              pin_memory=self.pin_memory
                              )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.cfg.hyperparameters.batch_size, shuffle=False, 
                        #   collate_fn=lambda b: collate_fn(b, input_pad_token_id=self.tokenizer.pad_token_id),
                          collate_fn=partial(partial_collate_fn, input_pad_token_id=self.tokenizer.pad_token_id),
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory
                          )
        

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.cfg.hyperparameters.batch_size, shuffle=False, 
                                    #  collate_fn=lambda b: collate_fn(b, input_pad_token_id=self.tokenizer.pad_token_id),
                                    collate_fn=partial(partial_collate_fn, input_pad_token_id=self.tokenizer.pad_token_id),
                                     num_workers=self.num_workers,
                                     pin_memory=self.pin_memory
                                     )
    

    