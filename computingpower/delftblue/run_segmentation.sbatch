#!/bin/sh
#SBATCH --job-name="segmentation_squared"
#SBATCH --time=10:00:00
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=2G
#SBATCH --account=research-eemcs-insy

module load 2022r2
module load python/3.8.12
module load miniconda3/4.12.0
module load openmpi/4.1.1
module load cuda/11.7
#module load py-torch/1.10.0-intel-mkl
#module load py-pillow/8.0.0

module load py-pip
#python -m pip install --upgrade --user numpy


unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"

### Comment out next lines after first run
#conda info --environments
#conda create -n forestbio
###

conda activate forestbio

### Comment out next few lines after first run
#conda install -c conda-forge pip
#conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
#conda install -c conda-forge segmentation-models-pytorch
#conda install -c conda-forge pillow
#conda install -c conda-forge rasterio
#conda install -c conda-forge pytorch-lightning
#conda install -c conda-forge torchgeo
###

#python ../../models/segmenter.py
python ../../models/test.py
conda deactivate