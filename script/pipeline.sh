#!/bin/bash

source ~/.bashrc
source activate act
export PYTHONPATH=`pwd`

python -u ./exp/pipelines/run_framework.py --config_fn $1 --seed $RANDOM --proj_dir $PYTHONPATH
