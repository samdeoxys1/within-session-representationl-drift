#!/bin/bash
#SBATCH --job-name=waveform
#SBATCH --partition=genx
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zz737@nyu.edu
#SBATCH --output="./slurm_jobs/waveform.out"

module purge
~/miniconda3/condabin/conda activate base
conda activate jax

python extract_waveform_script.py

