import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig

class MLMHead(nn.Module):
	
    def __init__(self, config):
        super().__init__()
        self.linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.gelu = nn.GELU()
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, x):
        x = self.linear(x)
        x = self.gelu(x)
        x = self.layer_norm(x)
        x = self.decoder(x)
        return x

class MaskedLanguageModel(nn.Module):
	    
    def __init__(self, pvp, args):
        super().__init__()
        self.config = AutoConfig.from_pretrained(args.model)
        self.encoder = AutoModel.from_pretrained(args.model)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model)
        self.mlm_head = MLMHead(self.config)
        self.pvp = pvp

        self.apply(self._init_weights)
        
    # https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py#L748 
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
    def forward(self, input_ids, attention_mask, token_type_ids, label=None):
        outputs = self.encoder(input_ids, attention_mask, token_type_ids)
        
        # logits : (batch_size, sequence_length, vocab_size)
        logits = self.mlm_head(outputs.last_hidden_state)
        
        # make mask for [MASK] token
        mask_indices = (input_ids == self.tokenizer.mask_token_id)
        mask_indices = torch.nonzero(mask_indices, as_tuple=True)
        
        # mlm_logits : (batch_size, vocab_size)
        mlm_logits = logits[mask_indices]
        
        # mlm logits -> cls logits : (batch_size, vocab_size) -> (batch_size, num_labels)
        cls_logits = self.pvp.convert_plm_logits_to_cls_logits(mlm_logits)
        
        if label is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(cls_logits, label.long())            
            return loss

        return cls_logits