from ocpmodels.common.relaxation.ase_utils import OCPCalculator
import ase.io
from ase.optimize import BFGS
from ase.build import fcc100, add_adsorbate, molecule
from ase.constraints import FixAtoms
from ase.calculators.emt import EMT

import os
import numpy as np

# odac23 does not work well?
checkpoint = "gndt_oc22_all_s2ef.pt"
checkpoint_dir = "./downloaded_checkpoints/oc22/"
checkpoint_path = checkpoint_dir + checkpoint

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
calc = OCPCalculator(checkpoint_path=checkpoint_path, trainer=trainer)
#calc = EMT()

# Set up the calculator
adslab.calc = calc

opt = BFGS(adslab, trajectory="test.traj")

opt.run(fmax=0.05, steps=100)
