#!/bin/bash
#SBATCH --job-name=NetworkML-test_sing
#SBATCH --partition=netsys
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

module load singularity
singularity exec --nv -B /scratch:/scratch \
    ~/ml-testing.sif \
    bash /scratch/yd2618/NetworkML/test_singularity.sh