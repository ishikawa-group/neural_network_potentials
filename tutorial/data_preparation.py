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

