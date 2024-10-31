from fairchem.core.models.model_registry import model_name_to_local_file
from fairchem.core.common.relaxation.ase_utils import OCPCalculator
from ase.build import fcc111, add_adsorbate
from ase.optimize import BFGS
import matplotlib.pyplot as plt
from ase.visualize import view
from ase.io import read

# checkpoint is downloaded by the following command
checkpoint_path = model_name_to_local_file("GemNet-OC-S2EFS-OC20+OC22", local_cache="./downloaded_checkpoints/")

mol = read("POSCAR")
mol.calc = OCPCalculator(checkpoint_path=checkpoint_path, cpu=True)

# Run the optimization
opt = BFGS(mol, trajectory="test.traj", logfile="test.log")
opt.run(fmax=0.05, steps=100)

# Visualize the result
view(opt)