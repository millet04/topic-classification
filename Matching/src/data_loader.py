import torch
import random
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import PreTrainedModel, AutoModel, AutoTokenizer, AutoConfig

class TopicClassificationDataset(Dataset):

    label_text = {
        0 : "IT/과학",
        1 : "경제",
        2 : "사회",
        3 : "생활문화",
        4 : "세계",
        5 : "스포츠",
        6 : "정치",
    }
    
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
                    
    def set_split(self, split='train', k=1):
        title = self._data_set[split]['title']
        label = self._data_set[split]['label']
        label_indices = list(TopicClassificationDataset.label_text.keys())

        title_lst, label_idx, label_txt, entailment = [], [], [], []
        
        for t, l in zip(title, label):
            title_temp = []
            label_idx_temp = []
            label_txt_temp = []
            entailment_temp = []
            
            if split == 'test':
                # all possible labels for test
                sampled_labels = label_indices               
            else:
                # negative sampling (with one positive)
                sampled_labels = self.sample_negative(label_indices, l, k)

            for s in sampled_labels:
                title_temp.append(t)
                label_txt_temp.append(TopicClassificationDataset.label_text[s])
                
                if l == s:
                    entailment_temp.append(0)
                    label_idx_temp.append(s)
                else:
                    entailment_temp.append(1)

            title_lst.append(title_temp)
            label_idx.append(label_idx_temp)
            label_txt.append(label_txt_temp)
            entailment.append(entailment_temp)
        
        self.title_lst = title_lst
        self.label_idx = label_idx
        self.label_txt = label_txt
        self.entailment = entailment
            
    def sample_negative(self, labels, positive, k):
        # negative sampling
        negatives = [label for label in labels if label != positive]
        sampled_negatives = random.sample(negatives, k)
        
        # add positive for training
        sampled_negatives.insert(0, positive)
        return sampled_negatives
    
    def __len__(self):
        assert len(self.title_lst) == len(self.label_idx)
        assert len(self.title_lst) == len(self.label_txt)
        assert len(self.title_lst) == len(self.entailment)
        return len(self.title_lst)
            
    def __getitem__(self, index):
        return {'title_lst':self.title_lst[index],
                'label_idx':self.label_idx[index],
                'label_txt':self.label_txt[index],
                'entailment':self.entailment[index]}


class CustomDataset(Dataset):

    label_text = {
        0 : "IT/과학",
        1 : "경제",
        2 : "사회",
        3 : "생활문화",
        4 : "세계",
        5 : "스포츠",
        6 : "정치",
    }
    
    def __init__(self, title_lst, label_idx, label_txt, entailment):
        self.title_lst = title_lst
        self.label_idx = label_idx
        self.label_txt = label_txt
        self.entailment = entailment   
            
    @classmethod
    def load_data(cls, file_path):
        data = pd.read_csv(file_path, delimiter='\t')
        title = list(data['title'])
        label = list(data['label'])
        label_indices = list(CustomDataset.label_text.keys())
        
        title_lst, label_idx, label_txt, entailment = [], [], [], []
        
        for t, l in zip(title, label):
            title_temp = []
            label_idx_temp = []
            label_txt_temp = []
            entailment_temp = []
        
            for i in label_indices:
                title_temp.append(t)
                label_txt_temp.append(CustomDataset.label_text[i])
                
                if l == i:
                    entailment_temp.append(0)
                    label_idx_temp.append(i)
                else:
                    entailment_temp.append(1)

            title_lst.append(title_temp)
            label_idx.append(label_idx_temp)
            label_txt.append(label_txt_temp)
            entailment.append(entailment_temp)       
            
        return cls(title_lst, label_idx, label_txt, entailment)
    
    def __len__(self):
        assert len(self.title_lst) == len(self.label_idx)
        assert len(self.title_lst) == len(self.label_txt)
        assert len(self.title_lst) == len(self.entailment)
        return len(self.title_lst)
            
    def __getitem__(self, index):
        return {'title_lst':self.title_lst[index],
                'label_idx':self.label_idx[index],
                'label_txt':self.label_txt[index],
                'entailment':self.entailment[index]}


class DataCollator(object):

    def __init__(self, args):
        self.tokenizer = AutoTokenizer.from_pretrained(args.model)
        self.template = args.template
        self.padding = args.padding          
        self.max_length = args.max_length   
        self.truncation = args.truncation  
    
    def __call__(self, samples):
        
        title_btx, label_btx, template_btx, entailment_btx = [], [], [], []        
        
        for sample in samples:
            title_btx.extend(sample['title_lst'])
            label_btx.extend(sample['label_idx'])
            entailment_btx.extend(sample['entailment'])
            for label_txt in sample['label_txt']:
                template_btx.append(self.template.format(label_txt))
        
        # positive pair : [cls] title [sep] label in template [sep]
        text_encode = self.tokenizer(
            title_btx,
            template_btx,
            padding = self.padding,
            truncation = self.truncation,
            max_length = self.max_length
            )
            
        return {
            'input_ids': torch.tensor(text_encode['input_ids']),
            'attention_mask': torch.tensor(text_encode['attention_mask']),            
            'token_type_ids':torch.tensor(text_encode['token_type_ids']),
            'entailment': torch.tensor(entailment_btx),
            'label': label_btx,
        }