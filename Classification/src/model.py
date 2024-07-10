import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig

class ClassifierHead(nn.Module):
	
    def __init__(self, num_labels, config):
        super().__init__()
        self.linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.bn = nn.BatchNorm1d(config.hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.gelu = nn.GELU()
        self.output = nn.Linear(config.hidden_size, num_labels) 
    
    def forward(self, x):
        x = self.linear(x)
        x = self.bn(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.output(x)
        return x

class Pooler(nn.Module):

    def __init__(self, pooler_type, config):
        super().__init__()
        self.pooler_type = pooler_type
        self.config = config
        
        assert self.pooler_type in ['pooler_output', 'cls', 'mean', 'max']
        
    def forward(self, outputs, attention_mask):
        last_hidden_state = outputs.last_hidden_state
        # hidden_states = outputs.hidden_states

        if self.pooler_type == 'pooler_output':
            return outputs.pooler_output
        
        elif self.pooler_type == 'cls':
            return last_hidden_state[:,0,:]
        
        # code from https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/models/Pooling.py#L9-L241 
        elif self.pooler_type == 'mean':
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            mean_embeddings = sum_embeddings / sum_mask  # mean_embeddings : (batch_size, hidden_size)
            return mean_embeddings 

        # code from https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/models/Pooling.py#L9-L241
        elif self.pooler_type == 'max':
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            last_hidden_state[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
            max_embeddings = torch.max(last_hidden_state, 1)[0] # max_embeddings : (batch_size, hidden_size)
            return max_embeddings
            
        else:
            raise NotImplementedError

class ClassificationModel(nn.Module):
	    
    def __init__(self, num_labels, args):
        super().__init__()
        self.config = AutoConfig.from_pretrained(args.model)
        self.encoder = AutoModel.from_pretrained(args.model)
        self.pooler = Pooler(args.pooler_type, self.config)
        self.classifier = ClassifierHead(num_labels, self.config)
            
    def forward(self, input_ids, attention_mask, token_type_ids, label=None):
        outputs = self.encoder(input_ids, attention_mask, token_type_ids)
        outputs = self.pooler(outputs, attention_mask)
        # logits : (batch_size, hidden_state)
        logits = self.classifier(outputs)
                
        if label is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, label.long())            
            return loss

        return logits