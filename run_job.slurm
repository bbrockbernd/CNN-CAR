#!/bin/sh
#
#SBATCH --job-name="test_job"
#SBATCH --partition=gpu
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=32G

module load 2022r1
module load gpu
module load cuda/11.1.1-zo3qpgx




srun python3 ./Model2D.py
