#!/bin/bash

eval "$(conda shell.bash hook)"

module load slurm
conda activate fusion

python3 exp_linear_early.py $1