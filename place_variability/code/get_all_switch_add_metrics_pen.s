#!/bin/bash
#SBATCH --job-name=swinfo
#SBATCH --partition=ccn
#SBATCH --array=0-71%25
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zz737@nyu.edu
#SBATCH --output="./slurm_jobs/%A_%a.out"

module purge
~/miniconda3/condabin/conda activate base
conda activate jax

ERROR_LOG=./slurm_jobs/get_all_switch_add_metrics_pen.out
TEST_MODE=0

python -W ignore get_all_switch_add_metrics_pen.py ${SLURM_ARRAY_TASK_ID} $TEST_MODE 2> >(grep -v -e '^[[:space:]]*$' | tee >(cat >&2) | (grep . && echo "====" ) >> $ERROR_LOG)

