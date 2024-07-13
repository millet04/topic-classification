import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig

class EntailmentHead(nn.Module):
	
    def __init__(self, config):
        super().__init__()
        self.linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.bn = nn.BatchNorm1d(config.hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.gelu = nn.GELU()
        self.output = nn.Linear(config.hidden_size, 2) 
    
    def forward(self, x):
        x = self.linear(x)    
        x = self.bn(x)         
        x = self.gelu(x)       
        x = self.dropout(x)   
        x = self.output(x)    
        return x

class EntailmentModel(nn.Module):
	    
    def __init__(self, args):
        super().__init__()
        self.config = AutoConfig.from_pretrained(args.model)
        self.encoder = AutoModel.from_pretrained(args.model)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model)
        self.entailment = EntailmentHead(self.config)
            
    def forward(self, input_ids, attention_mask, token_type_ids, entailment=None):
        outputs = self.encoder(input_ids, attention_mask, token_type_ids)
        cls = outputs.last_hidden_state[:,0,:]
        
        # logits : (batch_size, 2)
        logits = self.entailment(cls)

        if entailment is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, entailment.long())            
            return loss

        return logits