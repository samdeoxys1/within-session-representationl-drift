#!/bin/bash
#SBATCH --job-name=cosw_cofire
#SBATCH --partition=genx
#SBATCH --array=0-71%10
##SBATCH --array=0-13
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zz737@nyu.edu
#SBATCH --output="./slurm_jobs/%A_%a.out"

module purge
~/miniconda3/condabin/conda activate base
conda activate jax

ERROR_LOG=./slurm_jobs/co_sw_and_ripple_cofiring_one_sess.out
TEST_MODE=0

python -W ignore co_sw_and_ripple_cofiring_one_sess.py ${SLURM_ARRAY_TASK_ID} $TEST_MODE 2> >(grep -v -e '^[[:space:]]*$' | tee >(cat >&2) | (grep . && echo "====" ) >> $ERROR_LOG)
