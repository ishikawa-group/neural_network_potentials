from fairchem.core.preprocessing import AtomsToGraphs
from fairchem.core.datasets import LmdbDataset
import ase.io
from ase.build import bulk
from ase.build import fcc100, add_adsorbate, molecule
from ase.constraints import FixAtoms
from ase.calculators.emt import EMT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
# from ase.md.langevin import Langevin
from ase.md.verlet import VelocityVerlet
from ase import units
import matplotlib.pyplot as plt
import lmdb
import pickle
from tqdm import tqdm
import torch
import os

# Generate toy dataset: Relaxation of CO on Cu
adslab = fcc100("Cu", size=(2, 2, 3))
ads = molecule("CO")
add_adsorbate(adslab, ads, 3, offset=(1, 1))
cons = FixAtoms(indices=[atom.index for atom in adslab if (atom.tag == 3)])
adslab.set_constraint(cons)
adslab.center(vacuum=13.0, axis=2)
adslab.set_pbc(True)
adslab.set_calculator(EMT())

temperature_K = 300
steps = 100
traj_name = "CuCO_adslab.traj"

# dyn = BFGS(adslab, trajectory=traj_name, logfile=None)
# dyn.run(fmax=0, steps=steps)

MaxwellBoltzmannDistribution(adslab, temperature_K=temperature_K)
# dyn = Langevin(adslab, trajectory=traj_name, timestep=1.0*units.fs, temperature_K=temperature_K, friction=0.10/units.fs)
dyn = VelocityVerlet(adslab, trajectory=traj_name, timestep=1.0*units.fs)
dyn.run(steps=steps)

raw_data = ase.io.read(traj_name, ":")

a2g = AtomsToGraphs(
    max_neigh=50,
    radius=6,
    r_energy=True,    # False for test data
    r_forces=True,    # False for test data
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

dataset_name = "sample_CuCO.lmdb"

db = lmdb.open(
    dataset_name,
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

    # assign tags, if available
    data.tags = torch.LongTensor(tags)

    # Filter data if necessary
    # FAIRChem filters adsorption energies > |10| eV and forces > |50| eV/A

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

dataset = LmdbDataset({"src": dataset_name})
