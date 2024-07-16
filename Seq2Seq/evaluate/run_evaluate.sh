#!/bin/sh

GPU_ID=0
CUDA_VISIBLE_DEVICES=$GPU_ID python3 evaluate.py \
    	--model 'gogamza/kobart-base-v2' \
        --pretrained_path '../pretrained_model/tc_model'\
        --test_data '../data/test_data.csv'\
        --padding  \
        --truncation \