#!/bin/sh
#SBATCH --job-name="convertdata"
#SBATCH --time=10:00:00
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=2G
#SBATCH --account=research-eemcs-insy

module load 2022r2
module load python/3.8.12
module load miniconda3/4.12.0

module load py-pip
python -m pip install --upgrade --user numpy


unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"

### Comment out next lines after first run
conda info --environments
conda create -n forestbio
###

conda activate forestbio

### Comment out next few lines after first run
conda install -c conda-forge pip
###

python ../../utils/data_generator/get_train_data.py
conda deactivate