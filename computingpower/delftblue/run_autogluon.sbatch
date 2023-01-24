#!/bin/bash -l
#
#SBATCH --job-name="EpochIII"
#SBATCH --time=24:00:00
#SBATCH --partition=memory
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=400G
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
conda install -c default rasterio segmentation-models-pytorch pytorch-lightning torchgeo pillow gdal poppler
conda install pip
pip install autogluon.tabular[all]
### =======================================================

srun python ../../models/autogluon_train_cpu.py
conda deactivate
