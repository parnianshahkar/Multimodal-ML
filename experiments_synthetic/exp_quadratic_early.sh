#!/bin/bash

eval "$(conda shell.bash hook)"

conda activate fusion

python3 exp_quadratic_early.py $1 $2 $3