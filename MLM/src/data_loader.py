import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import PreTrainedModel, AutoModel, AutoTokenizer, AutoConfig

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
        self.set_split(split='train')
    
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


class DataCollator(object):

    def __init__(self, pvp, args):
        self.pvp = pvp
        self.tokenizer = AutoTokenizer.from_pretrained(args.model)
        self.padding = args.padding          
        self.max_length = args.max_length   
        self.truncation = args.truncation  
    
    def __call__(self, samples):
        
        title_lst, label_lst = [], []
        for sample in samples:
            title = self.pvp.truncate(sample['title'])
            title_lst.append(self.pvp.get_parts(title))
            label_lst.append(sample['label'])
        
        text_encode = self.tokenizer(
            title_lst,
            padding = self.padding,
            truncation = self.truncation,
            max_length = self.max_length
            )
            
        input_ids = torch.tensor(text_encode['input_ids'])
        attention_mask = torch.tensor(text_encode['attention_mask'])
        token_type_ids = torch.tensor(text_encode['token_type_ids'])        
        label = torch.tensor(label_lst)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,            
            'token_type_ids':token_type_ids,
            'label': label
        }