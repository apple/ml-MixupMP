#!/bin/bash

#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#


export netType='wide-resnet'
export depth=28
export width=10
export dataset='cifar10'

python ../main.py \
    --lr 0.1 \
    --net_type ${netType} \
    --depth ${depth} \
    --widen_factor ${width} \
    --dropout 0 \
    --dataset ${dataset} \
    --seed 30\
    --method 'mixupsoftv1' \
    --num_epochs 1
