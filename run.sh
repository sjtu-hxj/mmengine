#!/bin/bash
module purge
module load anaconda/2022.10
module load cuda/11.8

source activate hxj
export PYTHONUNBUFFERED=1
python train.py
