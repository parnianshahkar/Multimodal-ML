#!/bin/bash

eval "$(conda shell.bash hook)"

conda activate fusion

python3 exp_neuron.py $1 $2 $3