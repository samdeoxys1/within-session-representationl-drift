#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1-0:00:00
#SBATCH --mem=64GB
#SBATCH --partition=ccn
#SBATCH --array=0-74%5
#SBATCH --output="slurm_jobs/switch_analysis/%A_%a.out"


module purge
~/miniconda3/condabin/conda activate base
conda activate jax

python switch_analysis_one_session.py ${SLURM_ARRAY_TASK_ID}