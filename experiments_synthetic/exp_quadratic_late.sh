#!/bin/bash

eval "$(conda shell.bash hook)"

conda activate fusion

python3 exp_quadratic_late.py $1 $2 $3