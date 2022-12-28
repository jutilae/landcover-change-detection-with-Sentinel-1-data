#!/bin/bash
#SBATCH --job-name=raster_classification_tpot
#SBATCH --output=raster_classification_tpot_out_%A_%a.txt
#SBATCH --error=raster_classification_tpot_err_%A_%a.txt
#SBATCH --account=project_2004990
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=80G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1


module load pytorch
pip install numpy scipy scikit-learn pandas joblib --user
pip install deap update_checker tqdm stopit xgboost --user
pip install dask[delayed] dask[dataframe] dask-ml fsspec>=0.3.3 distributed>=2.10.0 --user
pip install tpot --usersa


# Feed the filename to the Python script
srun python 02_raster_classification_tpot_yearly_composites.py

