#!/bin/bash
#SBATCH --job-name=raster_classification
#SBATCH --output=array_job_out_%A_%a.txt
#SBATCH --error=array_job_err_%A_%a.txt
#SBATCH --account=project_2004990
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=80G
#SBATCH --partition=small
#SBATCH --array=1-2



module load geoconda



# Feed the filename to the Python script
srun python yearly_composites_raster_classification_only_SVM.py ${SLURM_ARRAY_TASK_ID}

