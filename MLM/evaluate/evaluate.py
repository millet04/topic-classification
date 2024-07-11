import os
import sys
import random
import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

sys.path.append("../") 
from src.data_loader import (
        TopicClassificationDataset,
        DataCollator,
)
from src.pvp import KluePVP
from src.model import MaskedLanguageModel
from utils import (
        get_accuracy,
        test,
)

def seed_everything(args):
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    os.environ["PYTHONHASHSEED"] = str(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def argument_parser():
    parser = argparse.ArgumentParser(description="Inference with KluePVP")
    
    parser.add_argument("--model", type=str, default="klue/bert-base",
                        help="Path to the model."
                       )
    parser.add_argument("--pretrained_path", type=str, default="./pretrained_model/tc_model",
                        help="Path to the model."
                       )
    parser.add_argument("--test_data", type=str, default="../data/test_data.csv",
                        help="Custom dataset to evaluate"
                       )
    parser.add_argument("--pattern", type=str, default='[MASK]:{}',
                        help="Pattern used for training."
                       )
    parser.add_argument('--batch_size', type=int, default=25,
                        help='Batch size'
                       )  
    parser.add_argument('--padding', action="store_true",
                        help='Add padding to short sentences'
                       )
    parser.add_argument('--truncation', action="store_true",
                        help='Truncate extra tokens when exceeding the max_length'
                       )
    parser.add_argument("--max_length", type=int, default=256,
                        help="Maximum length of text to be inferenced."
                       )
    parser.add_argument('--device', default = 'cuda' if torch.cuda.is_available() else 'cpu', type=str,
                        help = 'Choose a type of device for training'
                       )    
    parser.add_argument("--random_seed", type=int, default=42,
                        help="Random seed"
                       )
   
    args = parser.parse_args()
    return args

def main(args):
    seed_everything(args)
    
    dataset = TopicClassificationDataset.load_data()
    dataset.load_custom_testset(args.test_data)
    pvp = KluePVP(args)
    
    collator = DataCollator(pvp, args)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collator)

    model = MaskedLanguageModel(pvp, args).to(args.device)
    model.load_state_dict(torch.load(os.path.join(args.pretrained_path, "pytorch_model.bin")))

    accuracy = test(model, dataloader, args.device)
    print(f'Topic Classification Accuracy: {round(accuracy,2)} (%)')


if __name__ == '__main__':
    args = argument_parser()
    main(args)