#!/bin/sh
#SBATCH --job-name="convertdata"
#SBATCH --time=20:00:00
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8G
#SBATCH --account=Education-DDREAM-Epoch

module load 2022r2
module load python/3.8.12

module load py-pip
python -m pip install --upgrade --user numpy


unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"

### Comment out next lines after first run
conda info --environments
#conda create -n forestbiomassconda
###

conda activate forestbiomassconda

### Comment out next few lines after first run
#conda install -c conda-forge gdal
conda install -c conda-forge pip
pip install --global-option=build_ext --global-option="-I/usr/include/gdal" GDAL==`gdal-config --version`
#conda list
###

python ../../utils/data_generator/get_train_data.py
conda deactivate