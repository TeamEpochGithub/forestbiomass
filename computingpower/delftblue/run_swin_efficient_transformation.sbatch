#!/bin/bash -l
#
#SBATCH --job-name="EpochIII"
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gpus-per-task=2
#SBATCH --mem-per-cpu=2G
#SBATCH --account=research-eemcs-insy

module load 2022r2 miniconda3 py-pip/21.1.2
unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"

### ==============COMMENT OUT AFTER FIRST RUN==============
#conda create -n forest python=3.8.12
### =======================================================

conda activate forest

### ==============COMMENT OUT AFTER FIRST RUN==============
#conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia albumentations
#conda install -c conda-forge rasterio segmentation-models-pytorch pytorch-lightning torchgeo pillow gdal albumentations poppler
#conda install -c conda-forge albumentations
#conda remove opencv
#python -m pip install opencv-python
### =======================================================

srun python ../../models/segmenter_efficient_swin_transformation.py
conda deactivate