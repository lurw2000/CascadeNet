#!/bin/bash
#SBATCH --job-name=NetworkML
#SBATCH --partition=netsys
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=/gpfsnyu/scratch/rl4689/NetworkML/logs/cascadegan_sigcomm24/%j.out
#SBATCH --error=/gpfsnyu/scratch/rl4689/NetworkML/logs/cascadegan_sigcomm24/%j.err

module load anaconda3
source activate /gpfsnyu/home/rl4689/.conda/envs/ml-testing
cd ~/ML-testing-dev/demo/
time python training.py --path=$path
time python evaluating.py --path=$path
time python evaluating_timestamp_ratio.py --path=$path
time python evaluating-median_span.py --path=$path
time python evaluating-equal.py --path=$path
#time python generate_copy.py --path=$path