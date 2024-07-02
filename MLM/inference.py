import os
import random
import torch
import argparse
import numpy as np
from transformers import AutoTokenizer

from src.pvp import KluePVP
from src.model import MaskedLanguageModel

def seed_everything(args):
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    os.environ["PYTHONHASHSEED"] = str(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_default_args():
    parser = argparse.ArgumentParser(description="Inference with KluePVP")
    parser.add_argument("--model", type=str, default="klue/bert-base", help="Path to the model.")
    parser.add_argument("--pretrained_path", type=str, default="./pretrained_model/tc_model", help="Path to the model.")    
    parser.add_argument("--pattern", type=str, default='[MASK]:{}', help="Pattern used for training.")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum length of text to be inferenced.")
    parser.add_argument("--random_seed", type=int, default=42, help="")
    args, _ = parser.parse_known_args()
    return args

def inference(input_text):
    args = get_default_args()
    seed_everything(args)
    
    pvp = KluePVP(args)   
    
    model = MaskedLanguageModel(pvp, args)
    model.load_state_dict(torch.load(os.path.join(args.pretrained_path, "pytorch_model.bin")))
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    encode = tokenizer(pvp.get_parts(input_text), return_tensors='pt')
    
    model.eval()
    with torch.no_grad():
        logits = model(**encode) 

    logits_index = np.argmax(logits.detach().numpy(), axis=1).flatten()
    label = pvp.VERBALIZER[str(logits_index[0])][0] 

    return label

if __name__ == '__main__':
    while True:
        input_text = input('Enter text to be inferenced (type "exit" to quit): ')
        if input_text.lower() == "exit":
            print("Exiting the inference loop.")
            break
        result = inference(input_text)
        print(f"Prediction: {result}")