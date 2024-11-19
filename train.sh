#!/bin/bash

torchrun --nnodes 1 --nproc_per_node 8 train.py --nodes 1 --gpus_per_node 8