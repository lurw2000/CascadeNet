#!/bin/bash
#SBATCH --job-name=NetworkML-test
#SBATCH --partition=netsys
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --output=/gpfsnyu/scratch/yd2618/logs/test_conda/%j.out
#SBATCH --error=/gpfsnyu/scratch/yd2618/logs/test_conda/%j.err

module load anaconda3
source activate /gpfsnyu/home/yd2618/.conda/envs/eval
python -c 'import torch; print(torch.cuda.is_available())'