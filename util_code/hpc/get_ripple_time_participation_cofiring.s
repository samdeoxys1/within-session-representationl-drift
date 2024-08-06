#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=1-0:00:00
#SBATCH --mem=16GB
#SBATCH --partition=genx
#SBATCH --array=0-74%10 
#SBATCH --output="out/ripple/%A_%a.out"


module purge
~/miniconda3/condabin/conda activate base
conda activate jax

python get_ripple_time_participation_cofiring.py ${SLURM_ARRAY_TASK_ID}