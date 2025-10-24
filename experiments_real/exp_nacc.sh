#!/bin/bash

eval "$(conda shell.bash hook)"

module load slurm
conda activate fusion

python3 exp_nacc.py $1 $2 $3