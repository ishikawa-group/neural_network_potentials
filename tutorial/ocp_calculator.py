from ocpmodels.common.relaxation.ase_utils import OCPCalculator
from ase.calculators.emt import EMT
import ase.io
from ase.optimize import BFGS
from ase.build import fcc100, add_adsorbate, molecule
import os
import numpy as np
from ase.constraints import FixAtoms

adslab = fcc100("Cu", size=(3,3,3))
adsorbate = molecule("C3H8")
add_adsorbate(adslab, adsorbate, 3, offset=(1,1))
tags = np.zeros(len(adslab))
tags[18:27] = 1
tags[27:] = 2
adslab.set_tags(tags)
cons = FixAtoms(indices=[atom.index for atom in adslab if (atom.tag == 0)])
adslab.set_constraint(cons)
adslab.center(vacuum=13.0, axis=2)
adslab.set_pbc(True)
#ase.io.write("POSCAR", adslab);quit()

#config_yml_path = "configs/s2ef/all/gemnet/gemnet-dT.yml"
#checkpoint_path = "gemnet_t_direct_h512_all.pt"

config_yml_path = "configs/s2ef/all/schnet/schnet.yml"
checkpoint_path = "checkpoints/2023-11-14-09-42-40-SchNet-example/best_checkpoint.pt"

#calc = OCPCalculator(config_yml=config_yml_path, checkpoint_path=checkpoint_path)
calc = EMT()

adslab.calc = calc

#os.makedirs("data/sample_ml_relax", exist_ok=True)
opt = BFGS(adslab, trajectory="data/sample_ml_relax/toy_c3h8_relax_sch.traj")

opt.run(fmax=0.05, steps=100)

