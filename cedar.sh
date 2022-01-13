#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1              # Number of GPUs (per node)
#SBATCH --mem=8                  # memory (per node)
#SBATCH --time=0-1:00
#SBATCH --output=out.log          # output file
#SBATCH --error=err.log           # error file

python3.8 -u main.py