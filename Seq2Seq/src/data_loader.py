import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoConfig

class TopicClassificationDataset(Dataset):

    def __init__(self, train, valid, test):
        self.train = train
        self.valid = valid
        self.test = test
        self._data_set = {
            'train' : self.train,
            'valid' : self.valid,
            'test' : self.test,
            }
        
    @classmethod 
    def load_data(cls, test_size=0.1, seed=42):
        dataset = load_dataset('klue', 'ynat')
        train_valid = dataset['train'].train_test_split(test_size=test_size, seed=seed)
        train, valid = train_valid['train'], train_valid['test']
        test = dataset['validation']
        return cls(train, valid, test)
            
    def set_split(self, split='train'):
        self.title = self._data_set[split]['title']
        self.label = self._data_set[split]['label']

    def __len__(self):
        assert len(self.title) == len(self.label)
        return len(self.title)
            
    def __getitem__(self, index):
        return {'title':self.title[index],
                'label':self.label[index]}


class CustomDataset(Dataset):

    def __init__(self, title, label):
        self.title = title
        self.label = label
            
    @classmethod
    def load_data(cls, file_path):
        data = pd.read_csv(file_path, delimiter='\t')
        title = list(data['title'])
        label = list(data['label'])
        return cls(title, label)
    
    def __len__(self):
        assert len(self.title) == len(self.label)
        return len(self.title)
            
    def __getitem__(self, index):
        return {'title':self.title[index],
                'label':self.label[index]}


class DataCollator(object):

    def __init__(self, args):        
        self.tokenizer = AutoTokenizer.from_pretrained(args.model)
        self.tokenizer.add_special_tokens({'bos_token': '<s>'})
        self.padding = args.padding          
        self.max_length = args.max_length   
        self.truncation = args.truncation  
    
    def __call__(self, samples):
        
        label_lst, encoder_title_lst, decoder_title_lst = [], [], []
        for sample in samples:
            label_lst.append(sample['label'])
            encoder_title_lst.append(sample['title'])
            decoder_title_lst.append(self.tokenizer.bos_token + sample['title'] + self.tokenizer.eos_token)
            
        encoder_text_encode = self.tokenizer(
            encoder_title_lst,
            padding = self.padding,
            truncation = self.truncation,
            max_length = self.max_length
            )
        
        decoder_text_encode = self.tokenizer(
            decoder_title_lst,
            padding = self.padding,
            truncation = self.truncation,
            max_length = self.max_length
            )
        
        # label
        label = torch.tensor(label_lst)
        
        # Encoder input
        encoder_input_ids = torch.tensor(encoder_text_encode['input_ids'])
        encoder_attention_mask = torch.tensor(encoder_text_encode['attention_mask'])     
        
        # Decoder input
        decoder_input_ids = torch.tensor(decoder_text_encode['input_ids'])
        decoder_attention_mask = torch.tensor(decoder_text_encode['attention_mask'])
        
        return {
            'encoder_input_ids': encoder_input_ids,
            'encoder_attention_mask': encoder_attention_mask,            
            'decoder_input_ids':decoder_input_ids,
            'decoder_attention_mask': decoder_attention_mask,
            'label': label
        }