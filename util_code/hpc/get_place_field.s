#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1-0:00:00
#SBATCH --mem=8GB
#SBATCH --partition=genx
#SBATCH --array=0-74%25 
#SBATCH --output="saved_out/%A_%a.out"


module purge
~/miniconda3/condabin/conda activate base
conda activate jax

python get_place_field.py ${SLURM_ARRAY_TASK_ID}