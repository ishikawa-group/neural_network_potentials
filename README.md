# Using Open Catalyst Project
* This repository is to install, use, and do fine-tuning of Neural Network Potentials (NNPs) in Open Catalyst Project (OCP).
* Here, the procedures for installing, data generation, and fine-tuning are shown by step by step.

## Install
* The current setup uses python 3.11.X
* The original ocp repository is downloaded in the upper directory, but any directory is OK.

1. `git clone https://github.com/Open-Catalyst-Project/ocp.git ../ocp`
2. `pip install --upgrade pip`
3. `pip install torch==2.2.2`
4. `pip install lmdb==1.4.1 ase==3.22.1 pymatgen==2024.4.13 tensorboard==2.16.2 wandb==0.16.6`
5. `pip install torch_scatter torch_sparse torch_cluster torch_spline_conv torch_geometric -f https://data.pyg.org/whl/torch-2.2.2+cpu.html`
6. `cd ocp_original` then `pip install -e ../ocp`

## Download pre-trained checkpoint files
* Pre-trained checkpoints are available at OCP website.
  * https://open-catalyst-project.github.io/ocp/core/models.html

## Using NNP as ASE calculator
* The pre-trained NNPs can be used with the calculator in the Atomic Simulation Environment (ASE).
* Assuming that checkpoints are stored in `downloaded_checkpoints`.
* Sample script: `calculator.py`