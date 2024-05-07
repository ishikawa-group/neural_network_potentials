from ocpmodels.common.relaxation.ase_utils import OCPCalculator
import ase.io
from ase.optimize import BFGS
from ase.build import fcc100, add_adsorbate, molecule
from ase.constraints import FixAtoms
from ase.calculators.emt import EMT

import os
import numpy as np

#checkpoint_path = "./gemnet_t_direct_h512_all.pt"
#checkpoint_path = "/Users/ishi/opencatalystproject_new/checkpoints/2024-04-17-09-12-48-S2EF-example/best_checkpoint.pt"
#checkpoint = "gndt_oc22_all_s2ef.pt"
checkpoint = "schnet_all_large.pt"
checkpoint = "dimenetpp_all.pt"
checkpoint = "gemnet_t_direct_h512_all.pt"
checkpoint = "gemnet_oc_base_s2ef_all_md.pt"
checkpoint = "painn_h512_s2ef_all.pt"
checkpoint = "gndt_oc22_all_s2ef.pt"
#checkpoint = "gnoc_oc22_all_s2ef.pt"

# odac23 --- does not work well?

checkpoint_dir = "./downloaded_checkpoints/oc22/"
checkpoint_path = checkpoint_dir + checkpoint
#config_yml = "gemnet-dT.yml"
trainer = "forces"  # for S2EF

# Construct a sample structure
adslab = fcc100("Cu", size=(3, 3, 3))
adsorbate = molecule("C3H8")
add_adsorbate(adslab, adsorbate, 3, offset=(1, 1))
tags = np.zeros(len(adslab))
tags[18:27] = 1
tags[27:] = 2

adslab.set_tags(tags)
cons= FixAtoms(indices=[atom.index for atom in adslab if (atom.tag == 0)])
adslab.set_constraint(cons)
adslab.center(vacuum=13.0, axis=2)
adslab.set_pbc(True)

# Define the calculator
calc = OCPCalculator(checkpoint_path=checkpoint_path)
#calc = OCPCalculator(checkpoint_path=checkpoint_path, trainer=trainer)
#calc = OCPCalculator(checkpoint_path=checkpoint_path, config_yml=config_yml)
#calc = EMT()

# Set up the calculator
adslab.calc = calc

os.makedirs("test_traj", exist_ok=True)
opt = BFGS(adslab, trajectory="test_traj/test.traj")

opt.run(fmax=0.05, steps=100)

