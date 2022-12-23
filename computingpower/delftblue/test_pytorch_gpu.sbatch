#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=2
#SBATCH --mem-per-cpu=1G
#SBATCH --account=Education-DDREAM-Epoch


module load 2022r2 openmpi py-torch

srun python test_pytorch_gpu.py