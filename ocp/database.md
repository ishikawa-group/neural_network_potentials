# Making and using ASE datasets
* There are multiple ways to train and evaluate FAIRChem models on data other than OC20 and OC22.
* ASE-based dataset formats are also included as a convenience without using LMDBs.

## Using an ASE Database (ASE-DB)
* If your data is already in an ASE Database, no additional preprocessing is necessary before running training/prediction!
* If you want to effectively utilize more resources than this, consider writing your data to an LMDB.
* If your dataset is small enough to fit in CPU memory, use the `keep_in_memory: True` option to avoid this bottleneck.
* To use ASE-DB, we will just have to change our config files as

```yaml
dataset:
  format: ase_db
  train:
    src: # The path/address to your ASE DB
    connect_args:
      # Keyword arguments for ase.db.connect()
    select_args:
      # Keyword arguments for ase.db.select()
      # These can be used to query/filter the ASE DB
    a2g_args:
      r_energy: True
      r_forces: True
      # Set these if you want to train on energy/forces
      # Energy/force information must be in the ASE DB!
    keep_in_memory: False  # fast but only used for small datasets
    include_relaxed_energy: False  # Read the last structure's energy and save as "y_relaxed" for IS2RE
  val:
    src:
    a2g_args:
      r_energy: True
      r_forces: True
  test:
    src:
    a2g_args:
      r_energy: False
      r_forces: False
      # It is not necessary to have energy or forces when making predictions
```

## Using ASE-Readable Files
* It is possible to train/predict directly on ASE-readable files.
* This is only recommended for smaller datasets, as directories of many small files do not scale efficiently.
* There are two options for loading data with the ASE reader:

1. Single-Structure Files
* This dataset assumes a single structure will be obtained from each file.

```yaml
dataset:
  format: ase_read
  train:
    src: # The folder that contains ASE-readable files
    pattern: # Pattern matching each file you want to read (e.g. "*/POSCAR"). Search recursively with two wildcards: "**/*.cif".
    include_relaxed_energy: False # Read the last structure's energy and save as "y_relaxed" for IS2RE-Direct training

    ase_read_args:
      # Keyword arguments for ase.io.read()
    a2g_args:
      # Include energy and forces for training purposes
      # If True, the energy/forces must be readable from the file (ex. OUTCAR)
      r_energy: True
      r_forces: True
    keep_in_memory: False
```

2. Multi-structure Files
* This dataset supports reading files that each contain multiple structure (for example, an ASE.traj file).
* Using an index file, which tells the dataset how many structures each file contains, is recommended.
* Otherwise, the dataset is forced to load every file at startup and count the number of structures!

```yaml
dataset:
  format: ase_read_multi
  train:
    index_file: # Filepath to an index file which contains each filename and the number of structures in each file. e.g.:
            # /path/to/relaxation1.traj 200
            # /path/to/relaxation2.traj 150
            # ...
    # If using an index file, the src and pattern are not necessary
    src: # The folder that contains ASE-readable files
    pattern: # Pattern matching each file you want to read (e.g. "*.traj"). Search recursively with two wildcards: "**/*.xyz".

    ase_read_args:
      # Keyword arguments for ase.io.read()
    a2g_args:
      # Include energy and forces for training purposes
      r_energy: True
      r_forces: True
    keep_in_memory: False
```

# Making LMDB Datasets (original format, deprecated for ASE LMDBs)
* Storing your data in an LMDB ensures very fast random read speeds for the fastest supported throughput.
* This was the recommended option for the majority of fairchem use cases, but has since been deprecated for ASE LMDB files
* This notebook provides an overview of how to create LMDB datasets to be used with the FAIRChem repo.
* The corresponding Python script: `make_lmdb.py`

## Making dataset : An example of using EMT
```python
from fairchem.core.preprocessing import AtomsToGraphs
from fairchem.core.datasets import LmdbDataset
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

# Generate toy dataset: Relaxation of CO on Cu
adslab = fcc100("Cu", size=(2, 2, 3))
ads = molecule("CO")
add_adsorbate(adslab, ads, 3, offset=(1, 1))
cons = FixAtoms(indices=[atom.index for atom in adslab if (atom.tag == 3)])
adslab.set_constraint(cons)
adslab.center(vacuum=13.0, axis=2)
adslab.set_pbc(True)
adslab.set_calculator(EMT())

dyn = BFGS(adslab, trajectory="CuCO_adslab.traj", logfile=None)
dyn.run(fmax=0, steps=1000)
raw_data = ase.io.read("CuCO_adslab.traj", ":")
```

### Initialize AtomsToGraph feature extractor
* S2EF LMDBs utilize the TrajectoryLmdb dataset. This dataset expects a directory of LMDB files.
* We need to define `AtomsToGraph`. Its attributes are:
    + pos_relaxed: Relaxed adslab positions
    + sid: Unique system identifier, arbitrary
    + y_init: Initial adslab energy, formerly Data.y
    + y_relaxed: Relaxed adslab energy
    + tags (optional): 0 - subsurface, 1 - surface, 2 - adsorbate
	+ fid: Frame index along the trajcetory
* Additionally, a “length” key must be added to each LMDB file.
* 
```python
a2g = AtomsToGraphs(
    max_neigh=50,
    radius=6,
    r_energy=True,    # False for test data
    r_forces=True,    # False for test data
    r_distances=False,
    r_fixed=True,
)
```

### Initialize LMDB file
* Let's initialize the LMDB file, under some directory.
 
```python
os.makedirs("data/s2ef", exist_ok=True)

db = lmdb.open(
    "data/s2ef/sample_CuCO.lmdb",
    map_size=1099511627776*2,
    subdir=False,
    meminit=False,
    map_async=True,
)
```

## Write to LMDBs
* Now write the data in the trajectory file to LMDBs.

```python
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

dataset = LmdbDataset({"src": "s2ef/"})
```
