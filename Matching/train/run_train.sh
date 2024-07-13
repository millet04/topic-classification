#!/bin/sh

GPU_ID=0
CUDA_VISIBLE_DEVICES=$GPU_ID python3 train.py \
    	--model 'klue/bert-base' \
        --output_path '../pretrained_model/tc_model'\
        --template '이 기사는 {}에 관한 내용이다.'\
        --test_size 0.1\
    	--epochs 10 \
        --early_stop 5 \
        --batch_size 128 \
        --max_length 256 \
        --amp \
        --padding  \
        --truncation \
        --shuffle 