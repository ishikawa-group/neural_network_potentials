<<<<<<< HEAD
# opencatalystproject
=======
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
* We will use EMT for saving time.
```python
import os
import numpy as np
import ase.io
from ase.io.trajectory import Trajectory
from ase.io import extxyz
from ase.calculators.emt import EMT
from ase.build import fcc100, add_adsorbate, molecule
from ase.constraints import FixAtoms
from ase.optimize import LBFGS
from ase.visualize.plot import plot_atoms
from ase import Atoms

### DATA GENERATION

# This cell sets up and runs a structural relaxation 
# of a propane (C3H8) adsorbate on a copper (Cu) surface

adslab = fcc100("Cu", size=(3, 3, 3))
adsorbate = molecule("C3H8")
add_adsorbate(adslab, adsorbate, 3, offset=(1, 1)) # adslab = adsorbate + slab

# tag all slab atoms below surface as 0, surface as 1, adsorbate as 2
tags = np.zeros(len(adslab))
tags[18:27] = 1
tags[27:] = 2

adslab.set_tags(tags)

# Fixed atoms are prevented from moving during a structure relaxation. 
# We fix all slab atoms beneath the surface. 
cons= FixAtoms(indices=[atom.index for atom in adslab if (atom.tag == 0)])
adslab.set_constraint(cons)
adslab.center(vacuum=13.0, axis=2)
adslab.set_pbc(True)
adslab.set_calculator(EMT())

os.makedirs("data", exist_ok=True)

# Define structure optimizer - LBFGS. Run for 100 steps, 
# or if the max force on all atoms (fmax) is below 0 ev/A.
# fmax is typically set to 0.01-0.05 eV/A, 
# for this demo however we run for the full 100 steps.

dyn = LBFGS(adslab, trajectory="data/toy_c3h8_relax.traj")
dyn.run(fmax=0, steps=100)

traj = ase.io.read("data/toy_c3h8_relax.traj", ":")

# convert traj format to extxyz format (used by OC20 dataset)
columns = (["symbols", "positions", "move_mask", "tags"])

with open("data/toy_c3h8_relax.extxyz", "w") as f:
    extxyz.write_xyz(f, traj, columns=columns)


final_structure = traj[-1]
relaxed_energy = final_structure.get_potential_energy()
print(f'Relaxed absolute energy = {relaxed_energy} eV')

# Corresponding raw slab used in original adslab (adsorbate+slab) system.
raw_slab = fcc100("Cu", size=(3, 3, 3))
raw_slab.set_calculator(EMT())
raw_slab_energy = raw_slab.get_potential_energy()
print(f'Raw slab energy = {raw_slab_energy} eV')


adsorbate = Atoms("C3H8").get_chemical_symbols()
# For clarity, we define arbitrary gas reference energies here.
# A more detailed discussion of these calculations can be found in the corresponding paper's SI.
gas_reference_energies = {'H': .3, 'O': .45, 'C': .35, 'N': .50}

adsorbate_reference_energy = 0
for ads in adsorbate:
    adsorbate_reference_energy += gas_reference_energies[ads]

print(f'Adsorbate reference energy = {adsorbate_reference_energy} eV\n')

adsorption_energy = relaxed_energy - raw_slab_energy - adsorbate_reference_energy
print(f'Adsorption energy: {adsorption_energy} eV')
```

## Training
* Training NNP using the above dataset.
```python
from ocpmodels.datasets import TrajectoryLmdbDataset, SinglePointLmdbDataset

# TrajectoryLmdbDataset is our custom Dataset method to read the lmdbs as Data objects. 
# Note that we need to give the path to the folder containing lmdbs for S2EF
dataset = TrajectoryLmdbDataset({"src": "data/s2ef/train_100/"})

print("Size of the dataset created:", len(dataset))
print(dataset[0])
```

>>>>>>> master
