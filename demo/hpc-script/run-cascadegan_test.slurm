#!/bin/bash
#SBATCH --job-name=NetworkML
#SBATCH --partition=netsys
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4090:1
#SBATCH --output=/gpfsnyu/scratch/yd2618/logs/new_exp/%j.out
#SBATCH --error=/gpfsnyu/scratch/yd2618/logs/new_exp/%j.err

# Load necessary modules
module load anaconda3

# Activate the conda environment
source activate /gpfsnyu/home/yd2618/.conda/envs/cascadenet

# Print debugging information
echo "Job ID: $SLURM_JOB_ID"
echo "Running on $(hostname)"
echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"
nvidia-smi

cd /gpfsnyu/scratch/yd2618/ML-testing-dev/
# Execute the experiment script
bash config/experiments/figure_2_6_7_8_9_11_12_13_14_15_17_23_table_2.sh
bash config/experiments/figure_10_24_25_26_27_28.sh
bash config/experiments/figure_16.sh
bash config/experiments/figure_18.sh
bash config/experiments/figure_19.sh
bash config/experiments/figure_20_21_22.sh