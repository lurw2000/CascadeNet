#!/bin/bash
#SBATCH --job-name=ctgan
#SBATCH --partition=netsys
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4090:1
#SBATCH --output=/gpfsnyu/scratch/yd2618/logs/ctgan/%j.out
#SBATCH --error=/gpfsnyu/scratch/yd2618/logs/ctgan/%j.err

# Load necessary modules
module load anaconda3

# Activate the conda environment
source activate /gpfsnyu/home/yd2618/.conda/envs/ctgan

# Print debugging information
echo "Job ID: $SLURM_JOB_ID"
echo "Running on $(hostname)"
echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"
nvidia-smi

cd /gpfsnyu/scratch/yd2618/ML-testing-dev/baselines/ctgan

python training.py --input=../../data/caida/raw.csv --output=../../result/ctgan/caida --type=pcap
python evaluating.py --input=../../data/caida/raw.csv --output=../../result/ctgan/caida --type=pcap

python training.py --input=../../data/ca/raw.csv --output=../../result/ctgan/ca --type=pcap
python evaluating.py --input=../../data/ca/raw.csv --output=../../result/ctgan/ca --type=pcap

python training.py --input=../../data/dc/raw.csv --output=../../result/ctgan/dc --type=pcap
python evaluating.py --input=../../data/dc/raw.csv --output=../../result/ctgan/dc --type=pcap

python training.py --input=../../data/ton_iot/normal_1.csv --output=../../result/ctgan/ton --type=pcap
python evaluating.py --input=../../data/ton_iot/normal_1.csv --output=../../result/ctgan/ton --type=pcap