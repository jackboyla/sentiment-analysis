
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import re
import utils
import lightning as L
from sklearn.model_selection import train_test_split

import pandas as pd
from transformers import AutoTokenizer
from functools import partial
import html

run_logger = utils.create_logger(__name__)

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

def make_lowercase(text):
    return text.lower()

class TweetDataset(Dataset):
    def __init__(self, tweets, labels, tokenizer):
        self.tweets = tweets
        self.tweets = self.tweets.apply(convert_html_entities)
        self.tweets = self.tweets.apply(remove_urls)
        self.tweets = self.tweets.apply(remove_mentions_hashtags)
        # self.tweets = self.tweets.apply(make_lowercase)
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
    

class RandomDataset(Dataset):
    def __init__(self, size, num_samples, num_classes=2):
        self.len = num_samples
        assert num_samples % num_classes == 0, "num_samples % num_classes should = 0"
        sample_subset = num_samples // num_classes

        # self.data = np.random.randn(num_samples, size)
        self.data  = np.concatenate((np.full((num_samples, size), fill_value=6), 
                                     np.full((num_samples, size), fill_value=7),
                                     np.full((num_samples, size), fill_value=9)),
                                     axis=0)
        

        self.labels = np.concatenate((np.zeros(sample_subset), np.ones(sample_subset), np.full((num_samples), fill_value=2)))

    def __getitem__(self, index):
        input_dict = {'input_ids': self.data[index],
                      'attention_mask' : self.data[index],
                      'labels': self.labels[index]
                      }
        return input_dict

    def __len__(self):
        return self.len

    
def collate_fn(batch, input_pad_token_id=0.0):
    '''
    dynamically padding input sequences per batch
    batch is alist of dicts
    '''

    attention_masks = [torch.tensor(sample['attention_mask']) for sample in batch]
    input_ids = [torch.tensor(sample['input_ids']) for sample in batch]
    labels = [sample['labels'] for sample in batch]

    # Calculate the lengths of input sequences
    input_lengths = torch.tensor([len(seq) for seq in input_ids])

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=input_pad_token_id)
    attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=input_pad_token_id)

    # Sort the sequences, masks, and labels based on lengths in descending order
    sorted_lengths, sorted_indices = torch.sort(input_lengths, descending=True)
    input_ids = input_ids[sorted_indices]
    attention_masks = attention_masks[sorted_indices]
    labels = [labels[i] for i in sorted_indices]

    # Create a mask for padding tokens
    src_key_padding_mask = input_ids.eq(input_pad_token_id)  

    batch_dict = {'input_ids': input_ids.long(),
                  'attention_mask': attention_masks.float(),
                  'src_key_padding_mask': src_key_padding_mask,
                  'sorted_lengths': sorted_lengths, 
                  'sorted_indices': sorted_indices
                  }
        
    return batch_dict, torch.tensor(labels, dtype=torch.int64)


def partial_collate_fn(b, input_pad_token_id):
    return collate_fn(b, input_pad_token_id)

class TweetDataModule(L.pytorch.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_workers = self.cfg.hyperparameters.num_workers
        self.pin_memory = self.cfg.hyperparameters.pin_memory

        self.label_decode_map = {'negative': 0, 
                                 'neutral': 1,
                                 'positive': 2}


    def prepare_data(self):
        """
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html#prepare-data
        """

        # If we provide a dictionary of data files
        if type(self.cfg.datafiles.data_dirs) != str:
            df = pd.DataFrame()

            for file, label in self.cfg.datafiles.data_dirs.items():
                temp_df = pd.read_csv(file, sep='\t', **self.cfg.datafiles.get('kwargs', {})
                                    )
                if label in ['positive', 'neutral', 'negative']:
                    temp_df['label'] = label
                df = pd.concat([df, temp_df])

            def encode_sentiment(label):
                return self.label_decode_map[label]

            df['target'] = df['label'].apply(lambda x: encode_sentiment(x))
        else:
            DATASET_COLUMNS = ["target", "ids", "date", "flag", "user", "text"]
            self.label_decode_map = {0: 0, 
                                    4: 1}
            df = pd.read_csv(
                self.cfg.datafiles.data_dirs,
                encoding=self.cfg.datafiles.dataset_encoding,
                names=DATASET_COLUMNS
                )
            def binarize_sentiment(label):
                return self.label_decode_map[int(label)]

            df['target'] = df.target.apply(lambda x: binarize_sentiment(x))

        df.reset_index(inplace=True, drop=True)


        train_size = int(self.cfg.data_processing.train_size if self.cfg.data_processing.train_size > 1.0 else self.cfg.data_processing.train_size*len(df))
        val_size = int(self.cfg.data_processing.val_size if self.cfg.data_processing.val_size > 1.0 else self.cfg.data_processing.val_size*len(df))
        test_size = int(self.cfg.data_processing.test_size if self.cfg.data_processing.test_size > 1.0 else self.cfg.data_processing.test_size*len(df))
        print(f"Train/Val/Test Splits: {train_size}/{val_size}/{test_size}")

        self.train_df, self.test_df = train_test_split(df, train_size=train_size+val_size, test_size=test_size, random_state=42, stratify=df['target'])
        self.train_df, self.val_df = train_test_split(self.train_df, train_size=train_size, test_size=val_size, random_state=42, stratify=self.train_df['target'])

        self.train_df.reset_index(inplace=True, drop=True)
        self.val_df.reset_index(inplace=True, drop=True)
        self.test_df.reset_index(inplace=True, drop=True)

        self.tokenizer = self.get_tokenizer()


    def get_tokenizer(self):
        if 'transformers' in self.cfg.data_processing.tokenizer.object:
                    tokenizer = AutoTokenizer.from_pretrained(**self.cfg.data_processing.tokenizer.kwargs)
        else:
            tokenizer = utils.load_obj(self.cfg.data_processing.tokenizer.object)
            
            tokenizer = tokenizer(
                train_sequences=self.train_df['text'].values,
                **self.cfg.data_processing.tokenizer.get('kwargs', {})
                )
        return tokenizer


    def setup(self, stage: str):
        """
        Method to setup your datasets, here you can use whatever dataset class you have defined in Pytorch and prepare the data in order to pass it to the loaders later
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html#setup
        """
        if self.cfg.debug_dataflow:
            run_logger.info("Debug Random Dataset in use!")
            num_classes = 3
            num_samples = num_classes * 100
            seq_len = 32
            self.train_dataset = RandomDataset(seq_len, num_samples, num_classes=num_classes)
            self.val_dataset = RandomDataset(seq_len, num_samples, num_classes=num_classes)
            self.test_dataset = RandomDataset(seq_len, num_samples, num_classes=num_classes)
            self.sampler=None
            self.shuffle=True
        else:
            if stage == "fit":
                self.train_dataset = TweetDataset(self.train_df['text'], self.train_df['target'], self.tokenizer)
                self.val_dataset = TweetDataset(self.val_df['text'], self.val_df['target'], self.tokenizer)
                
            # Assign test dataset for use in dataloader(s)
            if stage == "test":
                self.test_dataset = TweetDataset(self.test_df['text'], self.test_df['target'], self.tokenizer)

            # Set up Random Weighted Sampling to account for class imbalance
            counts = self.train_df['label'].value_counts()
            class_weights = 1./counts
            samples_weight = np.array([class_weights[t] for t in self.train_df['target']])

            samples_weight = torch.from_numpy(samples_weight)
            samples_weight = samples_weight.double()
            self.sampler = WeightedRandomSampler(weights=samples_weight, 
                                                num_samples=len(samples_weight), 
                                                replacement=True)
            if self.sampler:
                self.shuffle=False


    def train_dataloader(self):
        return DataLoader(self.train_dataset, 
                            batch_size=self.cfg.hyperparameters.batch_size, 
                            sampler=self.sampler,
                            shuffle=self.shuffle, 
                            collate_fn=partial(partial_collate_fn, input_pad_token_id=self.tokenizer.pad_token_id),
                            num_workers=self.num_workers,
                            pin_memory=self.pin_memory
                            )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.cfg.hyperparameters.batch_size, shuffle=False, 
                        collate_fn=partial(partial_collate_fn, input_pad_token_id=self.tokenizer.pad_token_id),
                        num_workers=self.num_workers,
                        pin_memory=self.pin_memory
                        )
        

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.cfg.hyperparameters.batch_size, shuffle=False, 
                                    collate_fn=partial(partial_collate_fn, input_pad_token_id=self.tokenizer.pad_token_id),
                                    num_workers=self.num_workers,
                                    pin_memory=self.pin_memory
                                    )
    