import os
import re
import time
import random
import logging
import datetime
import argparse 
import numpy as np
from copy import deepcopy
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast 
from torch.cuda.amp import GradScaler
from transformers import AutoTokenizer, AutoConfig
from transformers import get_linear_schedule_with_warmup

import sys
sys.path.append("../") 

from src.data_loader import (
        TopicClassificationDataset,
        CustomDataset,
        DataCollator,
)  

from src.model import ClassificationModel
from evaluate.utils import (
        test,
        get_accuracy,
)

LOGGER = logging.getLogger()

def argument_parser():

    # Required
    parser = argparse.ArgumentParser(description='klue topic classification')

    parser.add_argument('--model', type=str, default = 'gogamza/kobart-base-v2',
                        help='Directory for pretrained model'
                       )    
    parser.add_argument('--output_path', type=str, default='../pretrained_model/tc_model',
                        help='Directory for output'
                       )
    parser.add_argument('--test_size', type=float, default=0.1,
                        help='Ratio of the test dataset'
                       )  
    
    # Tokenizer & Collator settings
    parser.add_argument('--max_length', default=32, type=int,
                        help='Max length of sequence'
                       )
    parser.add_argument('--padding', action="store_true",
                        help='Add padding to short sentences'
                       )
    parser.add_argument('--truncation', action="store_true",
                        help='Truncate extra tokens when exceeding the max_length'
                       )
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size'
                       )
    parser.add_argument('--shuffle', action="store_true",
                        help='Load shuffled sequences'
                       )

    # Train config    
    parser.add_argument('--epochs', default=1, type=int,
                        help='Training epochs'
                       )   
    parser.add_argument('--early_stop', default=5, type=int,
                        help='Early stop'
                       )   
    parser.add_argument('--weight_decay', default=1e-2, type=float,
                        help='Weight decay'
                       )       
    parser.add_argument('--no_decay', nargs='+', default=['bias', 'LayerNorm.weight'],
                        help='List of parameters to exclude from weight decay' 
                       )              
    parser.add_argument('--learning_rate', default=5e-5, type=float,
                        help='Leraning rate'
                       )       
    parser.add_argument('--eta_min', default=0, type=int,
                        help='Eta min for CosineAnnealingLR scheduler'
                       )   
    parser.add_argument('--eps', default=1e-8, type=float,
                        help='Epsilon for AdamW optimizer'
                       )   
    parser.add_argument('--amp', action="store_true",
                        help='Use Automatic Mixed Precision for training'
                       )  
    parser.add_argument('--device', default = 'cuda' if torch.cuda.is_available() else 'cpu', type=str,
                        help = 'Choose a type of device for training'
                       )
    parser.add_argument('--random_seed', default = 42, type=int,
                        help = 'Random seed'
                       )  
    args = parser.parse_args()
    return args

def init_logging():
    LOGGER.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]','%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    LOGGER.addHandler(console)


def seed_everything(args):
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    os.environ["PYTHONHASHSEED"] = str(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def format_time(elapsed):
    elapsed_rounded = int(round((elapsed))) # Round to the nearest second.
    return str(datetime.timedelta(seconds=elapsed_rounded)) # Format as hh:mm:ss


def get_adamw_optimizer(model, args):
    if args.no_decay: 
        # skip weight decay for some specific parameters i.e. 'bias', 'LayerNorm'.
        no_decay = args.no_decay  
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    else:
        # weight decay for every parameter.
        optimizer_grouped_parameters = model.parameters()
    optimizer = AdamW(optimizer_grouped_parameters, lr = args.learning_rate, eps = args.eps)
    return optimizer


def get_scheduler(optimizer, args):
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.eta_min, last_epoch=-1)
    return scheduler


def train(model, train_dataloader, optimizer, scheduler, args, scaler):
    total_train_loss = 0
    
    model.train()
    for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            
        # pass the data to device(cpu or gpu)            
        encoder_input_ids = batch['encoder_input_ids'].to(args.device)
        encoder_attention_mask = batch['encoder_attention_mask'].to(args.device)
        decoder_input_ids = batch['decoder_input_ids'].to(args.device)
        decoder_attention_mask = batch['decoder_attention_mask'].to(args.device)        
        label = batch['label'].to(args.device)

        optimizer.zero_grad()

        if args.amp:
            with autocast():
                train_loss = model(encoder_input_ids, encoder_attention_mask, decoder_input_ids, decoder_attention_mask, label)
        else:
            train_loss = model(encoder_input_ids, encoder_attention_mask, decoder_input_ids, decoder_attention_mask, label)
                
        total_train_loss += train_loss.mean()
                
        if args.amp:
            scaler.scale(train_loss.mean()).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            train_loss.mean().backward()
            optimizer.step()
            
        scheduler.step()
    
    avg_train_loss = total_train_loss / len(train_dataloader)
    return avg_train_loss


def valid(model, valid_dataloader, args):
    
    total_valid_loss = 0
    
    model.eval()    
    for _, batch in enumerate(valid_dataloader):
            
        encoder_input_ids = batch['encoder_input_ids'].to(args.device)
        encoder_attention_mask = batch['encoder_attention_mask'].to(args.device)
        decoder_input_ids = batch['decoder_input_ids'].to(args.device)
        decoder_attention_mask = batch['decoder_attention_mask'].to(args.device)   
        label = batch['label'].to(args.device)
            
        with torch.no_grad():
            valid_loss = model(encoder_input_ids, encoder_attention_mask, decoder_input_ids, decoder_attention_mask, label)
            total_valid_loss += valid_loss.mean()
    
    avg_valid_loss = total_valid_loss / len(valid_dataloader)
    return total_valid_loss  


def main(args):
    
    init_logging()
    seed_everything(args)
    
    LOGGER.info('*** KLUE Topic Classification ***')    
    
    collator = DataCollator(args)
    
    dataset = TopicClassificationDataset.load_data(test_size=args.test_size,seed=args.random_seed)
    dataset.set_split('train')
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size,
                                  shuffle=args.shuffle, collate_fn=collator)

    dataset = TopicClassificationDataset.load_data(test_size=args.test_size,seed=args.random_seed)
    dataset.set_split('valid')
    valid_dataloader = DataLoader(dataset, batch_size=args.batch_size,
                                  shuffle=args.shuffle, collate_fn=collator)

    dataset = TopicClassificationDataset.load_data(test_size=args.test_size,seed=args.random_seed)
    dataset.set_split('test')
    test_dataloader = DataLoader(dataset, batch_size=args.batch_size,
                                 shuffle=args.shuffle, collate_fn=collator)
                                
    num_labels = len(list(set(dataset.label)))
    
    if args.device == 'cuda':
        model = ClassificationModel(num_labels, args).to(args.device)
        model = torch.nn.DataParallel(model)
        LOGGER.info("Using nn.DataParallel")
    else:
        model = ClassificationModel(num_labels, args).to(args.device)

    optimizer = get_adamw_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args)

    if args.amp:
        scaler = GradScaler()
    else:
        scaler = None
    
    early_stop_loss = list()
    best_loss, best_model = None, None
    
    t0 = time.time()

    for epoch_i in range(args.epochs):
        
        LOGGER.info(f'Epoch : {epoch_i+1}/{args.epochs}')

        train_loss = train(model, train_dataloader, optimizer, scheduler, args, scaler)
        valid_loss = valid(model, valid_dataloader, args)
        
        # Check Best Model
        if not best_loss or valid_loss < best_loss:
            best_loss = valid_loss
            if isinstance(model, torch.nn.DataParallel):
                model_to_save = model.module   
            else: model_to_save = model
            best_model = deepcopy(model_to_save)
            
        # Early Stopping
        if len(early_stop_loss) == 0 or valid_loss > early_stop_loss[-1]:
            early_stop_loss.append(valid_loss)
            if len(early_stop_loss) == args.early_stop:break                                      
        else: early_stop_loss = list() 
                    
        print(f'Epoch:{epoch_i+1},Train_Loss:{round(float(train_loss.mean()), 4)},Valid_Loss:{round(float(valid_loss.mean()), 4)}') 
        
   # Save Best Model
    if not os.path.exists(args.output_path):
       os.makedirs(args.output_path)
    torch.save(best_model.state_dict(), os.path.join(args.output_path, "pytorch_model.bin"))
    LOGGER.info(f'>>> Saved Best Model at {args.output_path}')
    
    training_time = format_time(time.time() - t0)
    print(f'Total Training Time:  {training_time}')
    
    accuracy = test(best_model, test_dataloader, args.device)
    print(f'Topic Classification Accuracy: {round(accuracy, 2)} (%)')


if __name__ == '__main__':
    LOGGER = logging.getLogger()
    args = argument_parser()
    main(args)
