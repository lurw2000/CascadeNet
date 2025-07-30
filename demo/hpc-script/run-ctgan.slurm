#!/bin/bash
#SBATCH --job-name=NetworkML
#SBATCH --partition=netsys
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=logs/ctgan/%j.out
#SBATCH --error=logs/ctgan/%j.err

module load anaconda3
source activate /gpfsnyu/home/rl4689/.conda/envs/ctgan
cd ~/ML-testing-dev/evaluation/ctgan
time python training.py --input=${input} --output=${output} --type=${type}
time python evaluating.py --input=${input} --output=${output}