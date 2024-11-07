# OpenCatalystProject (Fairchem)
* This repository is to install, use, and do fine-tuning of Neural Network Potentials (NNPs) in Fairchem.
  * https://github.com/FAIR-Chem/fairchem
* The practical problem is as follows, and the tutorial for these problems are stored in their own directory.
  1. proton diffusion

## Modified manual 
* Following is the summary fairchem manual page.
* Here, we only treat the **S2EF** mode, as it is the simplest.

## Installation
1. Install pytorch
2. Install torch_geometric, torch_scatter, torch_sparse, and torch_cluster
	* see PyG website
3. Install fairchem-core
```bash
git clone https://github.com/FAIR-Chem/fairchem.git
pip install fairchem/packages/fairchem-core
```
* Note: `fairchem` is under active development. Please keep the library up-to-date.
