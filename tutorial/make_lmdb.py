from ocpmodels.preprocessing import AtomsToGraphs
from ocpmodels.datasets import SinglePointLmdbDataset, TrajectoryLmdbDataset
import ase.io
from ase.build import bulk
from ase.build import fcc100, add_adsorbate, molecule
from ase.constraints import FixAtoms
from ase.calculators.emt import EMT
from ase.optimize import BFGS
import matplotlib.pyplot as plt
import lmdb
import pickle
from tqdm import tqdm
import torch
import os

adslab = fcc100("Cu", size=(2, 2, 3))
ads = molecule("CO")
add_adsorbate(adslab, ads, 3, offset=(1, 1))
cons = FixAtoms(indices=[atom.index for atom in adslab if (atom.tag == 3)])
adslab.set_constraint(cons)
adslab.center(vacuum=13.0, axis=2)
adslab.set_pbc(True)
adslab.set_calculator(EMT())
dyn = BFGS(adslab, trajectory="CuCO_adslab.traj", logfile=None)
dyn.run(fmax=0, steps=100)

raw_data = ase.io.read("CuCO_adslab.traj", ":")

a2g = AtomsToGraphs(
    max_neigh=50,
    radius=6,
    r_energy=True,    # False for test data
    r_forces=True,    # False for test data
    r_distances=False,
    r_fixed=True,
)

os.makedirs("s2ef", exist_ok=True)
db = lmdb.open(
    "s2ef/sample_CuCO.lmdb",
    map_size=1099511627776 * 2,
    subdir=False,
    meminit=False,
    map_async=True,
)

tags = raw_data[0].get_tags()
data_objects = a2g.convert_all(raw_data, disable_tqdm=True)


for fid, data in tqdm(enumerate(data_objects), total=len(data_objects)):
    #assign sid
    data.sid = torch.LongTensor([0])
    
    #assign fid
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

dataset = TrajectoryLmdbDataset({"src": "s2ef/"})
len(dataset)

print(dataset[0])

