import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from transformers.models.bart.modeling_bart import BartEncoder, BartDecoder

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

class ClassificationModel(nn.Module):
	    
    def __init__(self, num_labels, args):
        super().__init__()
        self.config = AutoConfig.from_pretrained(args.model)
        self.seq2seq = AutoModel.from_pretrained(args.model)
        self.encoder = self.seq2seq.encoder
        self.decoder = self.seq2seq.decoder
        self.classifier = ClassifierHead(num_labels, self.config)
            
    def forward(self, encoder_input_ids, encoder_attention_mask, decoder_input_ids, decoder_attention_mask, label=None):
        
        encoder_outputs = self.encoder(input_ids = encoder_input_ids,
                                       attention_mask = encoder_attention_mask)
        
        decoder_outputs = self.decoder(input_ids = decoder_input_ids,
                                       attention_mask = decoder_attention_mask,
                                       encoder_hidden_states = encoder_outputs.last_hidden_state,
                                       encoder_attention_mask = encoder_attention_mask)
                
        hidden_states = decoder_outputs.last_hidden_state
        
        # https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/models/bart/modeling_bart.py#L94
        eos_mask = decoder_input_ids.eq(self.config.eos_token_id).to(hidden_states.device)
        if len(torch.unique_consecutive(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        sentence_representation = hidden_states[eos_mask, :].view(hidden_states.size(0), -1, hidden_states.size(-1))[:,-1,:]

        # logits : (batch_size, hidden_state)
        logits = self.classifier(sentence_representation)
                
        if label is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, label.long())            
            return loss

        return logits