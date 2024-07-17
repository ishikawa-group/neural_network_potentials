# Using Fairchem (previously OpenCatalystProject) neutral network potentials
* This repository is to install, use, and do fine-tuning of Neural Network Potentials (NNPs) in Fairchem.
  * https://github.com/FAIR-Chem/fairchem
* The practical problem is as follows, and the tutorial for these problems are stored in their own directory.
  1. proton diffusion

---

# Modified manual 
* Following is the summary fairchem manual page.
* Here, we only treat the **S2EF** mode, as it is the simplest.

# Installation
1. Install pytorch
2. Install torch_geometric, torch_scatter, torch_sparse, and torch_cluster
	* see PyG website
3. Install fairchem-core
```bash
git clone https://github.com/FAIR-Chem/fairchem.git
pip install fairchem/packages/fairchem-core
```
* Note: `fairchem` is under active development. Please keep the library up-to-date.

---

# Using pre-trained models in ASE
* In this section, the usage of NNP(neural network potential) under ASE is shown.
* The corresponding Python script: `calculate.py`

1. See what pre-trained potentials are available

```python
from fairchem.core.models.model_registry import available_pretrained_models
print(available_pretrained_models)  # you can see the available models
```

2. Choose a checkpoint you want to use and download it automatically: e.g. GemNet-OC, trained on OC20 and OC22.

```python
from fairchem.core.models.model_registry import model_name_to_local_file
from fairchem.core.common.relaxation.ase_utils import OCPCalculator
from ase.build import fcc111, add_adsorbate
from ase.optimize import BFGS
import matplotlib.pyplot as plt
from ase.visualize import view

# checkpoint is downloaded by the following command
checkpoint_path = model_name_to_local_file("GemNet-OC-S2EFS-OC20+OC22", local_cache="./downloaded_checkpoints/")

# Define the model atomic system, a Pt(111) slab with an *O adsorbate!
slab = fcc111('Pt', size=(2, 2, 5), vacuum=10.0)
add_adsorbate(slab, 'O', height=1.2, position='fcc')

# Load the pre-trained checkpoint!
calc = OCPCalculator(checkpoint_path=checkpoint_path, cpu=False)
slab.set_calculator(calc)

# Run the optimization
opt = BFGS(slab, trajectory="test.traj")
opt.run(fmax=0.05, steps=100)

# Visualize the result
view(opt)
```

---

# Available Pretrained models
* Pretrained NNP models should be found in the following website:
	* https://fair-chem.github.io/core/model_checkpoints.html

---

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

---

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

---

# Training and evaluating custom models on OCP datasets
## Getting Started
* This section is a tutorial for training and evaluating models for each S2EF.
* `main.py` serves as the entry point to run any task. This script requires two command line arguments at a minimum:
	1. `--mode MODE`: MODE can be train, predict or run-relaxations 
		to train a model, make predictions using an existing model, or run relaxations using an existing model, respectively.
	2. `--config-yml PATH`: PATH is the path to a YAML configuration file.
        The configs directory contains a number of example config files.
* Running `main.py` directly runs the model on a single CPU or GPU:
  * `python main.py --mode train --config-yml configs/TASK/SIZE/MODEL/MODEL.yml`
* The test case is already prepared: `python main.py --mode train --config-yml ./schnet.yml`
* Training results are stored in the log directory `logs/tensorboard/[TIMESTAMP]` where `[TIMESTAMP]` is starting time.
* You can see the training result by `tensorboard --logdir logs/tensorboard/[TIMESTAMP]`

### OC20
#### Structure to Energy and Forces (S2EF)
* In the S2EF task, the model takes the positions of the atoms as input and predicts the adsorption energy and per-atom forces.
* To train a model for the S2EF, you can use the OCPTrainer and TrajectoryLmdb dataset by specifying your configuration file:

```yaml
trainer: ocp

dataset:
  # Training data
  train:
    src: [Path to training data]
    normalize_labels: True
    # Mean and standard deviation of energies
    target_mean: -0.7586356401443481
    target_std: 2.981738567352295
    # Mean and standard deviation of forces
    grad_target_mean: 0.0
    grad_target_std: 2.981738567352295
  # Val data (optional)
  val:
    src: [Path to validation data]
  # Test data (optional)
  test:
    src: [Path to test data]
```

* You can find examples configuration files in `configs/s2ef`
* The checkpoint is stored in `checkpoints/[TIMESTAMP]/checkpoint.pt`
* Next, run this model on the test data: `python main.py --mode predict --config-yml configs/s2ef/2M/schnet/schnet.yml \
        --checkpoint checkpoints/[TIMESTAMP]/checkpoint.pt`
* The predictions are stored in `[RESULTS_DIR]/ocp_predictions.npz`

### Training OC20 models with total energies (S2EF)
* To train/validate an OC20 S2EF model on total energies instead of adsorption energies, you need to change the config file.
* They include setting as follows:
  + `dataset: oc22_lmdb`
  + `prediction_dtype: float32`
  + `train_on_oc20_total_energies: True`
  + `oc20_ref: path/to/oc20_ref.pkl`
* Also, please note that our evaluation server does not currently support OC20 total energy models.

```yaml
task:
  prediction_dtype: float32
  # ...

dataset:
  format: oc22_lmdb
  train:
    src: data/oc20/s2ef/train
    normalize_labels: False
    train_on_oc20_total_energies: True
    # download at https://dl.fbaipublicfiles.com/opencatalystproject/data/oc22/oc20_ref.pkl
    oc20_ref: path/to/oc20_ref.pkl
  val:
    src: data/oc20/s2ef/val_id
    train_on_oc20_total_energies: True
    oc20_ref: path/to/oc20_ref.pkl
```

## OC22
### Structure to Total Energy and Forces (S2EF-Total)
* The S2EF-Total task takes a structure and predicts the total DFT energy and per-atom forces.
* This differs from the original OC20 S2EF task because it predicts total energy instead of adsorption energy.
* To train an OC22 S2EF-Total model, you need the OC22LmdbDataset by including these lines in your configuration file:

```yaml
dataset:
  format: oc22_lmdb # Use the OC22LmdbDataset
  ...
```

* You can find examples configuration files in `configs/oc22/s2ef`.

## Available pre-trained models
### OC20
* SchNet-S2EF-OC20-200k, SchNet-S2EF-OC20-2M, SchNet-S2EF-OC20-20M, SchNet-S2EF-OC20-All
* SpinConv-S2EF-OC20-2M, SpinConv-S2EF-OC20-All
* GemNet-dT-S2EF-OC20-2M, GemNet-dT-S2EF-OC20-All
* PaiNN-S2EF-OC20-All
* GemNet-OC-S2EF-OC20-2M, GemNet-OC-S2EF-OC20-All, GemNet-OC-S2EF-OC20-All+MD, GemNet-OC-Large-S2EF-OC20-All+MD
* SCN-S2EF-OC20-2M, SCN-t4-b2-S2EF-OC20-2M, SCN-S2EF-OC20-All+MD
* eSCN-L4-M2-Lay12-S2EF-OC20-2M, eSCN-L6-M2-Lay12-S2EF-OC20-2M, eSCN-L6-M2-Lay12-S2EF-OC20-All+MD, eSCN-L6-M3-Lay20-S2EF-OC20-All+MD
* EquiformerV2-83M-S2EF-OC20-2M, EquiformerV2-31M-S2EF-OC20-All+MD, EquiformerV2-153M-S2EF-OC20-All+MD
* SchNet-S2EF-force-only-OC20-All
* DimeNet++-force-only-OC20-All, DimeNet++-Large-S2EF-force-only-OC20-All, DimeNet++-S2EF-force-only-OC20-20M+Rattled, DimeNet++-S2EF-force-only-OC20-20M+MD
* CGCNN-IS2RE-OC20-10k, CGCNN-IS2RE-OC20-100k, CGCNN-IS2RE-OC20-All
* DimeNet-IS2RE-OC20-10k, DimeNet-IS2RE-OC20-100k, DimeNet-IS2RE-OC20-all
* SchNet-IS2RE-OC20-10k, SchNet-IS2RE-OC20-100k, SchNet-IS2RE-OC20-All
* DimeNet++-IS2RE-OC20-10k, DimeNet++-IS2RE-OC20-100k, DimeNet++-IS2RE-OC20-All
* PaiNN-IS2RE-OC20-All

### OC22
* GemNet-dT-S2EFS-OC22, GemNet-OC-S2EFS-OC22, GemNet-OC-S2EFS-OC20+OC22, GemNet-OC-S2EFS-nsn-OC20+OC22, GemNet-OC-S2EFS-OC20->OC22 
* EquiformerV2-lE4-lF100-S2EFS-OC22

### ODAC
* SchNet-S2EF-ODAC
* DimeNet++-S2EF-ODAC
* PaiNN-S2EF-ODAC
* GemNet-OC-S2EF-ODAC
* eSCN-S2EF-ODAC
* EquiformerV2-S2EF-ODAC, EquiformerV2-Large-S2EF-ODAC
* Gemnet-OC-IS2RE-ODAC
* eSCN-IS2RE-ODAC
* EquiformerV2-IS2RE-ODAC

## Troubleshoting
* `CUDA out of memory. Tried to allocate X.XX GiB. GPU`
  * Reduce the batch size.