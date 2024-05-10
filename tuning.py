from ocpmodels.trainers import OCPTrainer
from ocpmodels.datasets import LmdbDataset
from ocpmodels import models
from ocpmodels.common import logger
from ocpmodels.common.utils import setup_logging, setup_imports
setup_logging()
setup_imports()

import numpy as np
import copy
import os

train_src = "data/s2ef/mytrain"
val_src   = "data/s2ef/myval"

train_dataset = LmdbDataset({"src": train_src})

energies = []
for data in train_dataset:
    energies.append(data.energy)

mean = np.mean(energies)
stdev = np.std(energies)

# Task
task = {
    'dataset': 'lmdb', # dataset used for the S2EF task
    'description': 'Regressing to energies and forces for DFT trajectories from OCP',
    'type': 'regression',
    'metric': 'mae',
    'labels': ['potential energy'],
    'grad_input': 'atomic forces',
    'train_on_free_atoms': True,
    'eval_on_free_atoms': True
}
# Model
model = {
  "name": "gemnet_oc",
  "num_spherical": 7,
  "num_radial": 128,
  "num_blocks": 4,
  "emb_size_atom": 64,
  "emb_size_edge": 64,
  "emb_size_trip_in": 64,
  "emb_size_trip_out": 64,
  "emb_size_quad_in": 32,
  "emb_size_quad_out": 32,
  "emb_size_aint_in": 64,
  "emb_size_aint_out": 64,
  "emb_size_rbf": 16,
  "emb_size_cbf": 16,
  "emb_size_sbf": 32,
  "num_before_skip": 2,
  "num_after_skip": 2,
  "num_concat": 1,
  "num_atom": 3,
  "num_output_afteratom": 3,
  "cutoff": 12.0,
  "cutoff_qint": 12.0,
  "cutoff_aeaint": 12.0,
  "cutoff_aint": 12.0,
  "max_neighbors": 30,
  "max_neighbors_qint": 8,
  "max_neighbors_aeaint": 20,
  "max_neighbors_aint": 1000,
  "rbf": {
    "name": "gaussian"
  },
  "envelope": {
    "name": "polynomial",
    "exponent": 5
  },
  "cbf": {"name": "spherical_harmonics"},
  "sbf": {"name": "legendre_outer"},
  "extensive": True,
  "output_init": "HeOrthogonal",
  "activation": "silu",

  "regress_forces": True,
  "direct_forces": True,
  "forces_coupled": False,

  "quad_interaction": True,
  "atom_edge_interaction": True,
  "edge_atom_interaction": True,
  "atom_interaction": True,

  "num_atom_emb_layers": 2,
  "num_global_out_layers": 2,
  "qint_tags": [1, 2],

  "scale_file": "ocp/configs/s2ef/all/gemnet/scaling_factors/gemnet-oc.pt",
}

# Optimizer
optimizer = {
    'batch_size': 1,         # originally 32
    'eval_batch_size': 1,    # originally 32
    'num_workers': 0,
    'lr_initial': 5.e-4,
    'optimizer': 'AdamW',
    'optimizer_params': {"amsgrad": True},
    'scheduler': "ReduceLROnPlateau",
    'mode': "min",
    'factor': 0.8,
    'patience': 3,
    'max_epochs': 1,         # used for demonstration purposes
    'force_coefficient': 100,
    'ema_decay': 0.999,
    'clip_grad_norm': 10,
    'loss_energy': 'mae',
    'loss_force': 'l2mae',
}
# Dataset
dataset = [
    {"src": train_src,
        "normalize_labels": True,
        "target_mean": mean,
        "target_std": stdev,
        "grad_target_mean": 0.0,
        "grad_target_std": stdev
    }, # train set
    {"src": val_src}
]

trainer = OCPTrainer(
    task=task,
    model=copy.deepcopy(model), # copied for later use, not necessary in practice.
    dataset=dataset,
    optimizer=optimizer,
    outputs={},
    loss_fns={},
    eval_metrics={},
    name="s2ef",
    identifier="S2EF-example",
    run_dir=".", # directory to save results if is_debug=False. Prediction files are saved here so be careful not to override!
    is_debug=False, # if True, do not save checkpoint, logs, or results
    print_every=5,
    seed=0, # random seed to use
    logger="tensorboard", # logger of choice (tensorboard and wandb supported)
    local_rank=0,
    amp=True, # use PyTorch Automatic Mixed Precision (faster training and less memory usage),
)

# train the model
trainer.train()
