from fairchem.core.models import model_name_to_local_file
from fairchem.core.common.relaxation.ase_utils import OCPCalculator
from ase import Atoms
from ase.db import connect
from ase.calculators.singlepoint import SinglePointCalculator
from fairchem.core.common.tutorial_utils import train_test_val_split
from fairchem.core.common.tutorial_utils import generate_yml_config
from fairchem.core.common.tutorial_utils import fairchem_main
import subprocess
import time
import json
import numpy as np
import matplotlib.pyplot as plt

# when clean
# subprocess.run("rm -rf ./checkpoints/*", shell=True)

# pretrained_model = "GemNet-OC-S2EFS-OC20+OC22"
# pretrained_model = "GemNet-dT-S2EFS-OC22"  # good
pretrained_model = "PaiNN-S2EF-OC20-All"

checkpoint_path = model_name_to_local_file(model_name=pretrained_model, local_cache="./checkpoints")

with open("supporting-information.json", "rb") as f:
    d = json.loads(f.read())

oxides = list(d.keys())
polymorphs = list(d["TiO2"].keys())

c = d["TiO2"]["rutile"]["PBE"]["EOS"]["calculations"][0]
atoms = Atoms(symbols=c["atoms"]["symbols"],
              positions=c["atoms"]["positions"],
              cell=c["atoms"]["cell"],
              pbc=c["atoms"]["pbc"])
atoms.set_tags(np.ones(len(atoms)))

calc = OCPCalculator(checkpoint_path=checkpoint_path, trainer="forces", cpu=False)

t0 = time.time()

eos_data = {}
for oxide in oxides:
    print(f"oxide: {oxide}")
    eos_data[oxide] = {}
    for polymorph in polymorphs:
        dft = []
        ocp = []
        vols = []
        calculations = d[oxide][polymorph]["PBE"]["EOS"]["calculations"]
        for c in calculations:
            atoms = Atoms(symbols=c["atoms"]["symbols"],
                          positions=c["atoms"]["positions"],
                          cell=c["atoms"]["cell"],
                          pbc=c["atoms"]["pbc"])
            atoms.set_tags(np.ones(len(atoms)))
            atoms.calc = calc
            ocp += [atoms.get_potential_energy()/len(atoms)]
            dft += [c["data"]["total_energy"]/len(atoms)]
            vols += [atoms.get_volume()]

        plt.plot(dft, ocp, marker="s" if oxide == "VO2" else ".", alpha=0.5, label=f"{oxide}-{polymorph}")
        eos_data[oxide][polymorph] = (vols, dft, ocp)

plt.xlabel("DFT (eV/atom)")
plt.ylabel("OCP (eV/atom)")
plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), ncols=3)

mae = np.mean(np.abs(np.array(dft) - np.array(ocp)))
print(f"MAE = {mae:1.3f} eV/atom")

db = connect("oxides_.db")

for oxide in oxides:
    for polymorph in polymorphs:
        for c in d[oxide][polymorph]["PBE"]["EOS"]["calculations"]:
            atoms = Atoms(symbols=c["atoms"]["symbols"],
                          positions=c["atoms"]["positions"],
                          cell=c["atoms"]["cell"],
                          pbc=c["atoms"]["pbc"])
            atoms.set_tags(np.ones(len(atoms)))
            calc = SinglePointCalculator(atoms, energy=c["data"]["total_energy"], forces=c["data"]["forces"])
            atoms.set_calculator(calc)
            db.write(atoms)

subprocess.run("rm -rf train.db test.db val.db *.db.lock", shell=True)
train, test, val = train_test_val_split("oxides_.db")

yml = "config.yml"
subprocess.run(["rm", yml])
generate_yml_config(checkpoint_path=checkpoint_path, yml=yml,
                    # delete=["slurm", "cmd", "logger", "task", "model_attributes", "optim.loss_force",
                    #        "dataset", "test_dataset", "val_dataset"],
                    delete=["slurm", "cmd", "logger", "task", "model_attributes",
                            "dataset", "test_dataset", "val_dataset"],
                    update={"gpus": 0,
                            "task.dataset": "ase_db",
                            "optim.eval_every": 10,
                            "optim.max_epochs": 1,
                            "optim.num_workers": 1,
                            "optim.batch_size": 16,
                            "logger": "tensorboard",

                            "dataset.train.src": "train.db",
                            "dataset.train.a2g_args.r_energy": True,
                            "dataset.train.a2g_args.r_forces": True,

                            "dataset.test.src": "test.db",
                            "dataset.test.a2g_args.r_energy": False,
                            "dataset.test.a2g_args.r_forces": False,

                            "dataset.val.src": "val.db",
                            "dataset.val.a2g_args.r_energy": True,
                            "dataset.val.a2g_args.r_forces": True,
                            }
                    )

print(f"config yaml file seved to {yml}.")

t0 = time.time()
subprocess.run(f"python main.py --mode train --config-yml {yml} --checkpoint {checkpoint_path} > train.txt 2>&1", shell=True)
print(f"Elapsed time = {time.time() - t0:1.1f} seconds")

cpline  = subprocess.check_output(["grep", "checkpoint_dir", "train.txt"])
cpdir   = cpline.decode().strip().replace(" ", "").split(":")[-1]
newchk  = cpdir + "/checkpoint.pt"
newcalc = OCPCalculator(checkpoint_path=newchk, cpu=True)

for oxide in oxides:
    print(f"oxide: {oxide}")
    for polymorph in polymorphs:
        dft = []
        ocp = []
        calculations = d[oxide][polymorph]["PBE"]["EOS"]["calculations"]
        for c in calculations:
            atoms = Atoms(symbols=c["atoms"]["symbols"],
                          positions=c["atoms"]["positions"],
                          cell=c["atoms"]["cell"],
                          pbc=c["atoms"]["pbc"])
            atoms.set_tags(np.ones(len(atoms)))
            atoms.calc = newcalc
            ocp += [atoms.get_potential_energy()/len(atoms)]
            dft += [c["data"]["total_energy"]/len(atoms)]

        plt.plot(dft, ocp, marker="s" if oxide == "VO2" else ".", alpha=0.5, label=f"{oxide}-{polymorph}")

plt.xlabel("DFT (eV/atom)")
plt.ylabel("OCP (eV/atom)")
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), ncol=3)
plt.show()

mae = np.mean(np.abs(np.array(dft) - np.array(ocp)))

print(f"New MAE = {mae:1.3f} eV/atom")
