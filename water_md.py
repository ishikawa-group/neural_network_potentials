from ocpmodels.common.relaxation.ase_utils import OCPCalculator
from ase.calculators.emt import EMT
from ase.io import read
#from ase.md.verlet import VelocityVerlet
from ase.md.nvtberendsen import NVTBerendsen
from ase import units
import os
import numpy as np
from ase.constraints import FixAtoms

mol = read("water_box.pdb")
mol.cell = [20, 20, 20]
mol.pbc = True

config_yml_path = "/Users/ishi/opencatalystproject/my_setup/configs/models/oc20/painn_h512.yml"
checkpoint_path = "/Users/ishi/opencatalystproject/my_setup/checkpoints/models/oc20/painn_h512_s2ef_all.pt"

calc = OCPCalculator(config_yml=config_yml_path, checkpoint_path=checkpoint_path)
#calc = EMT()

mol.calc = calc

dyn = NVTBerendsen(mol, timestep=1.0*units.fs, temperature=300, taut=1000*units.fs, trajectory="water.traj")
dyn.run(500)

