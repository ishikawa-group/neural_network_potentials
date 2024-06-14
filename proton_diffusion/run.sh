#!/bin/sh
#$ -cwd
#$ -l cpu_4=1
#$ -l h_rt=0:10:00
#$ -N serial

# loading CUDA
module load cuda
# loading Intel compiler
module load intel

python diffusion.py --checkpoint "./checkpoints/2024-06-14-10-33-52/checkpoint.pt"

