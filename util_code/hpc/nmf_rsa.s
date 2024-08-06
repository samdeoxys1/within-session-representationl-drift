#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1-0:00:00
#SBATCH --mem=64GB
#SBATCH --partition=ccn


module purge
# Load in what we need to execute mpirun.
module load slurm python/3.8.12

python nmf_rsa.py 0