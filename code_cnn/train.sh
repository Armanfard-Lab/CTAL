#!/bin/sh

python main.py \
    --model_file configs/cityscapes/hrnet18/m_ctal.yml \
    --storage_root logs \
    --cuda 0 --seed 0 --augmentation True
