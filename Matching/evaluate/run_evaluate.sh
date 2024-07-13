#!/bin/sh

GPU_ID=0
CUDA_VISIBLE_DEVICES=$GPU_ID python3 evaluate.py \
    	--model 'klue/bert-base' \
        --pretrained_path '../pretrained_model/tc_model'\
        --test_data '../data/test_data.csv'\
        --template '이 기사는 {}에 관한 내용이다.' \
        --padding  \
        --truncation \