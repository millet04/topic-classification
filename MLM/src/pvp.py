# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Portions of this file are based on work by Timo Schick
# Modifications by: Kim Min-Seok
# Date: 2024-07-14
# Description of changes: Modified the `pvp.py` file from the original code.

import torch
from transformers import AutoTokenizer

class KluePVP(object):
    
    VERBALIZER = {
        "0": ["과학"],
        "1": ["경제"],
        "2": ["사회"],
        "3": ["생활"],
        "4": ["세계"],
        "5": ["스포츠"],
        "6": ["정치"]
        }

    def __init__(self, args):
        self.pattern = args.pattern
        self.tokenizer = AutoTokenizer.from_pretrained(args.model)
        self.max_length = self.tokenizer.model_max_length
        if args.max_length:
            self.max_length = args.max_length
        self.label_list = list(KluePVP.VERBALIZER.keys())
                
    @property
    def max_num_verbalizers(self):
        return max(len(self.verbalize(label)) for label in self.label_list) 
        
    def _get_verbalizer_ids(self, verbalizer):
        verbalizer_id = self.tokenizer.convert_tokens_to_ids(verbalizer)
        assert verbalizer_id != self.tokenizer.unk_token_id, "Verbalizer was tokenized as <UNK>"        
        return verbalizer_id     
    
    def _count_tokens(self, example):
        return len(self.tokenizer.tokenize(example))

    def _build_mlm_logits_to_cls_logits_tensor(self):
        m2c_tensor = torch.ones([len(self.label_list), self.max_num_verbalizers], dtype=torch.long) * -1
        for label_idx, label in enumerate(self.label_list):
            verbalizers = self.verbalize(label)
            for verbalizer_idx, verbalizer in enumerate(verbalizers):
                verbalizer_id = self._get_verbalizer_ids(verbalizer)
                m2c_tensor[label_idx, verbalizer_idx] = verbalizer_id
        return m2c_tensor

    def _convert_single_mlm_logits_to_cls_logits(self, logits):
        m2c = self._build_mlm_logits_to_cls_logits_tensor().to(logits.device)
        # filler_len.shape() == max_fillers
        filler_len = torch.tensor([len(self.verbalize(label)) for label in self.label_list], dtype=torch.float)
        filler_len = filler_len.to(logits.device)

        # cls_logits.shape() == num_labels x max_fillers  (and 0 when there are not as many fillers).
        cls_logits = logits[torch.max(torch.zeros_like(m2c), m2c)]
        cls_logits = cls_logits * (m2c > 0).float()

        # cls_logits.shape() == num_labels
        cls_logits = cls_logits.sum(axis=1) / filler_len
        return cls_logits    

    def convert_plm_logits_to_cls_logits(self, logits):
        cls_logits = torch.stack([self._convert_single_mlm_logits_to_cls_logits(lgt) for lgt in logits])
        return cls_logits
    
    def get_parts(self, example):
        if self.tokenizer.mask_token not in self.pattern:
            raise Exception("Mask token must be included in the Pattern.")
        if '{}' not in self.pattern:
            raise Exception("Blank \'{}\' must be included in the Pattern.")
        prompt = self.pattern.format(example)       
        return prompt     
    
    def truncate(self, example):
        pat_len = self._count_tokens(self.pattern) - 2 # the count of {, }
        seq_len = self._count_tokens(example) + 2      # the count of [cls] and [seq]
        if pat_len + seq_len > self.max_length:
            extra = (pat_len + seq_len) - self.max_length 
            example = example[:-extra]
        return example
        
    def verbalize(self, label):
        return KluePVP.VERBALIZER[label] 
