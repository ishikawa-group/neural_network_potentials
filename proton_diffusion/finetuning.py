import os
# os.environ['OPENBLAS_NUM_THREADS'] = '1'  # if openblas error happens, uncomment this
from fairchem.core.models import model_name_to_local_file
from fairchem.core.common.relaxation.ase_utils import OCPCalculator
from ase import Atoms
from ase.db import connect
from ase.calculators.singlepoint import SinglePointCalculator
from fairchem.core.common.tutorial_utils import train_test_val_split
from fairchem.core.common.tutorial_utils import generate_yml_config
from fairchem.core.common.tutorial_utils import fairchem_main
import os
import subprocess
import time
import json
import matplotlib.pyplot as plt

# when clean checkpoint file
subprocess.run("rm -rf ./checkpoints/*", shell=True)

# pretrained_model = "DimeNet++-S2EF-ODAC"  # bad
# pretrained_model = "PaiNN-S2EF-OC20-All"
# pretrained_model = "GemNet-OC-S2EFS-OC22"
# pretrained_model = "GemNet-OC-S2EFS-OC20+OC22"  # takes time

checkpoint_path = model_name_to_local_file(model_name=pretrained_model, local_cache="../pretrained_checkpoints")

use_asedb = False  # when using ASE database

if use_asedb:
    subprocess.run("rm -rf train.db test.db val.db *.db.lock", shell=True)
    train, test, val = train_test_val_split("bulk.db")  # when using ASE database

yml = "config.yml"
subprocess.run(["rm", yml])

# --- training and validation data are always necessary!
generate_yml_config(checkpoint_path=checkpoint_path, yml=yml,
                    delete=["slurm", "cmd", "logger", "task",  # "model_attributes",
                            "dataset", "test_dataset", "val_dataset"],
                    update={"gpus": 0,
                            "trainer": "ocp",

                            # "eval_metrics.primary_metric": "forces_mae",

                            # "model.cutoff": 12.0,   # do not change
                            # "model.use_pbc": True,  # always make this True.

                            # "task.dataset": "ase_db",
                            "task.dataset": "lmdb",
                            "optim.eval_every": 1,
                            "optim.max_epochs": 1,
                            "optim.num_workers": 0,
                            "optim.batch_size": 20,  # number of samples in one batch ... maybe

                            "logger": "tensorboard",

                            # "dataset.train.src": "train.db",
                            "dataset.train.src": "../data/s2ef/mytrain",
                            "dataset.train.a2g_args.r_energy": True,
                            "dataset.train.a2g_args.r_forces": True,

                            # "dataset.test.src": "test.db",
                            # "dataset.test.a2g_args.r_energy": False,
                            # "dataset.test.a2g_args.r_forces": False,

                            # "dataset.val.src": "val.db",
                            "dataset.val.src": "../data/s2ef/myval",
                            "dataset.val.a2g_args.r_energy": True,
                            "dataset.val.a2g_args.r_forces": True,
                            }
                    )

print(f"config yaml file seved to {yml}.")

t0 = time.time()
subprocess.run(f"python ../main.py --mode train --config-yml {yml} --checkpoint {checkpoint_path} &> train.txt", shell=True)
print(f"Elapsed time = {time.time() - t0:1.1f} seconds")

cpline  = subprocess.check_output(["grep", "checkpoint_dir", "train.txt"])
cpdir   = cpline.decode().strip().replace(" ", "").split(":")[-1]
newchk  = cpdir + "/checkpoint.pt"

print(f"new checkpoint: {os.path.abspath(newchk)}")
