#!/bin/bash -l
#
#SBATCH --job-name="tensorflow"
#SBATCH --time=05:00:00
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=4
#SBATCH --mem-per-cpu=1G
#SBATCH --account=research-eemcs-insy

module load 2022r2 openmpi py-torch py-pillow miniconda3


unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"

###
conda create -n forest
###

conda activate forest

###
conda install -c conda-forge segmentation-models-pytorch
conda install -c conda-forge pytorch-lightning
conda install -c conda-forge rasterio
conda install -c conda-forge torchgeo
###

srun python ../../models/test.py
conda deactivate