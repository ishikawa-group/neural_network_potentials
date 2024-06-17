#!/bin/sh
#$ -cwd
#$ -l node_o=1
#$ -l h_rt=1:00:00

# loading CUDA
module load cuda
# loading Intel compiler
module load intel

pip uninstall -y torch torch_geometric torch_scatter torch_sparse
pip install -r ../requirements.txt

maxtime_ps=1.0

python make_lmdb_from_outcar.py >& out1.txt
python finetuning.py >& out2.txt
python diffusion.py --maxtime_ps ${maxtime_ps} >& out3.txt

