#!/bin/bash -l
#
#SBATCH --job-name="dist_training"
#SBATCH --time=05:00:00
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=64G
#SBATCH --gpus-per-task=1


module load 2022r1
module load gpu
module load python/3.8.12-bohr45d
module load openmpi
module load py-tensorflow




srun ./chief.sh > chief.txt &
srun ./tuner.sh tuner0 > tuner.txt &
srun ./tuner.sh tuner1 > tuner.txt &
srun ./tuner.sh tuner2 > tuner.txt &

