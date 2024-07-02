#!/bin/sh

GPU_ID=0
CUDA_VISIBLE_DEVICES=$GPU_ID python3 train.py \
    	--model 'klue/bert-base' \
        --output_path '../pretrained_model/tc_model'\
        --test_size 0.1\
        --pattern '[MASK]:{}'\
    	--epochs 10 \
        --batch_size 128 \
        --max_length 256 \
        --amp \
        --padding  \
        --truncation \
        --shuffle 