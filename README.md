# Open Catalyst Project
* This repository is to install, use, and do fine-tuning of Neural Network Potentials (NNPs) in Open Catalyst Project (OCP).
* The OCP repository:
* Here, the procedures for installing, data generation, and fine-tuning are shown by step by step.

## Install
1. `git clone https://github.com/Open-Catalyst-Project/ocp.git`
2. `pip install --upgrade pip`
3. `pip install torch==2.0.1`
4. `pip install lmdb==1.1.1 ase==3.21 pymatgen==2020.12.31 pyyaml==6.0.1 tensorboard==2.15.1 wandb==0.16.0`
5. `pip install torch_scatter torch_sparse torch_cluster torch_spline_conv torch_geometric -f https://data.pyg.org/whl/torch-2.0.1+cpu.html`
6. `pip install ipython orjson numba e3nn protobuf`
7. `cd ocp_original` thne `pip install -e .`

### download the training data (if you don't have)
```bash
cd tutorial/data
wget -q http://dl.fbaipublicfiles.com/opencatalystproject/data/tutorial_data.tar.gz -O tutorial_data.tar.gz
tar -xzvf tutorial_data.tar.gz
rm tutorial_data.tar.gz
```
## Data preparation
* Python script: `tutorial/data_generation.py`
* We will use EMT for saving time.

## Training
* Training NNP using the above dataset.
* Python script: `tutorial/training.py`

