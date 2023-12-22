from ocpmodels.common.relaxation.ase_utils import OCPCalculator
import ase.io
from ase.optimize import BFGS
from ase.md.verlet import VelocityVerlet
from ase.md.nvtberendsen import NVTBerendsen
from ase.build import fcc100, add_adsorbate, molecule
import os
import numpy as np
from ase.constraints import FixAtoms
from ase import units

vacuum = 13.0

adslab = fcc100("Cu", size=(3, 3, 4))
adsorbate = molecule("CO")
add_adsorbate(adslab, adsorbate, 3, offset=(1, 1))
cons = FixAtoms(indices=[atom.index for atom in adslab if (atom.tag > 2)])
adslab.set_constraint(cons)
adslab.center(vacuum=vacuum, axis=2)
adslab.translate([0, 0, -vacuum+0.1])
adslab.set_pbc(True)

config_yml_path = "configs/s2ef/all/gemnet/gemnet-dT.yml"
#config_yml_path = "configs/s2ef/all/painn/painn_h512.yml"

checkpoint_dir  = "2023-12-04-01-12-48-S2EF-gemnet"
checkpoint_path = "checkpoints/" + checkpoint_dir + "/best_checkpoint.pt"

calc = OCPCalculator(config_yml=config_yml_path, checkpoint_path=checkpoint_path)

adslab.calc = calc

opt = BFGS(adslab, trajectory="cu_surface.traj")
opt.run(fmax=0.05, steps=100)

#dyn = VelocityVerlet(adslab, timestep=1.0*units.fs, trajectory="cu_surface.traj", logfile="md_log.txt")
#dyn = NVTBerendsen(adslab, timestep=1.0*units.fs, temperature=300, taut=1000*units.fs, trajectory="cu_surface.traj", logfile="md_log.txt")
#dyn.run(steps=100)

