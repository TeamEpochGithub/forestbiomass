#!/bin/bash -l
#
#SBATCH --job-name="EpochIII"
#SBATCH --time=80:00:00
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --account=research-eemcs-insy

module load 2022r2 miniconda3
unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"

### ==============COMMENT OUT AFTER FIRST RUN==============
#conda create -n forest python=3.8.12
### =======================================================

conda activate forest

### ==============COMMENT OUT AFTER FIRST RUN==============
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
conda install -c conda-forge rasterio segmentation-models-pytorch pytorch-lightning torchgeo pillow gdal autogluon poppler
### =======================================================
pip3 install --pre autogluon

srun python ../../models/autogluon_train_cpu_philip.py
conda deactivate