#!/bin/sh

python main.py \
    --model_file configs/pascal/swinv2/ctal.yml \
    --storage_root logs \
    --cuda 0 --seed 0
