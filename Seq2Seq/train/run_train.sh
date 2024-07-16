#!/bin/sh

GPU_ID=0
CUDA_VISIBLE_DEVICES=$GPU_ID python3 train.py \
    	--model 'gogamza/kobart-base-v2' \
        --output_path '../pretrained_model/tc_model'\
        --test_size 0.1\
    	--epochs 10 \
        --early_stop 5 \
        --batch_size 128 \
        --max_length 32 \
        --amp \
        --padding  \
        --truncation \
        --shuffle 