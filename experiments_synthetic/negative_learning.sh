#!/bin/bash

eval "$(conda shell.bash hook)"

conda activate fusion

python3 negative_learning.py $1 $2