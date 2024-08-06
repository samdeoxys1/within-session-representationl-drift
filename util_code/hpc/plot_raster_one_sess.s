#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1-0:00:00
#SBATCH --mem=8GB
#SBATCH --partition=ccn
#SBATCH --array=0-74


module purge
module load slurm python/3.8.12
source ~/miniconda3/bin/activate neural

python plot_raster_one_sess.py ${SLURM_ARRAY_TASK_ID}