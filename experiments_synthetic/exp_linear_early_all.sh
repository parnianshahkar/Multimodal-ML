#!/bin/bash

eval "$(conda shell.bash hook)"

module load slurm
conda activate fusion

python3 exp_linear_early_all.py $1