import ase.io
from ase.build import fcc100, add_adsorbate, molecule
from ase.constraints import FixAtoms
from ase.calculators.emt import EMT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.langevin import Langevin
from ase.io import Trajectory
from ase import units
from ocpmodels.preprocessing import AtomsToGraphs
from ocpmodels.datasets import SinglePointLmdbDataset, TrajectoryLmdbDataset, LmdbDataset

import lmdb
import pickle
import torch
import os
import numpy as np
from tqdm import tqdm

# Construct a sample structure
slab = fcc100("Cu", size=(3, 3, 4))
adsorbate = molecule("CO")
add_adsorbate(slab, adsorbate, height=2.5, offset=(1, 1))
tags = np.zeros(len(slab))
tags[18:] = 1

slab.set_tags(tags)
cons= FixAtoms(indices=[atom.index for atom in slab if (atom.tag == 0)])
slab.set_constraint(cons)
slab.center(vacuum=13.0, axis=2)
slab.set_pbc(True)

# Define the calculator
calc = EMT()
slab.calc = calc

temperature_K = 500
MaxwellBoltzmannDistribution(slab, temperature_K=temperature_K)
dyn = Langevin(slab, timestep=1.0*units.fs, temperature_K=temperature_K, friction=0.03/units.fs)
trajectory_name = "test.traj"
traj = Trajectory(trajectory_name, "w", slab)
dyn.attach(traj.write, interval=1)

dyn.run(steps=200)

raw_data = ase.io.read(trajectory_name, ":")
print(len(raw_data))

a2g = AtomsToGraphs(
    max_neigh=50,
    radius=6,
    r_energy=True,    # False for test data
    r_forces=True,
    r_distances=False,
    r_fixed=True,
)

db = lmdb.open(
    "sample_CuCO.lmdb",
    map_size=1099511627776*2,
    subdir=False,
    meminit=False,
    map_async=True,
)

tags = raw_data[0].get_tags()
data_objects = a2g.convert_all(raw_data, disable_tqdm=True)

for fid, data in tqdm(enumerate(data_objects), total=len(data_objects)):
    # assign sid
    data.sid = torch.LongTensor([0])

    # assign fid
    data.fid = torch.LongTensor([fid])

    #assign tags, if available
    data.tags = torch.LongTensor(tags)

    # Filter data if necessary
    # OCP filters adsorption energies > |10| eV and forces > |50| eV/A

    # no neighbor edge case check
    if data.edge_index.shape[1] == 0:
        print("no neighbors", traj_path)
        continue

    txn = db.begin(write=True)
    txn.put(f"{fid}".encode("ascii"), pickle.dumps(data, protocol=-1))
    txn.commit()

txn = db.begin(write=True)
txn.put(f"length".encode("ascii"), pickle.dumps(len(data_objects), protocol=-1))
txn.commit()

db.sync()
db.close()

