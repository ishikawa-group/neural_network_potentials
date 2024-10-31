from fairchem.core.models.model_registry import model_name_to_local_file
from fairchem.core.common.relaxation.ase_utils import OCPCalculator
from ase.build import fcc111, add_adsorbate
from ase.optimize import BFGS
from ase.visualize.plot import plot_atoms
import matplotlib.pyplot as plt

checkpoint_path = model_name_to_local_file("EquiformerV2-31M-S2EF-OC20-All+MD", local_cache="./downloaded_checkpoints")
calc = OCPCalculator(checkpoint_path=checkpoint_path, cpu=True)

re1 = -3.03

slab = fcc111("Pt", size=(2,2,5), vacuum=10.0)
add_adsorbate(slab, "O", height=1.2, position="fcc")

slab.set_calculator(calc)
opt = BFGS(slab)
opt.run(fmax=0.05, steps=100)

slab_e = slab.get_potential_energy()
print(slab_e + re1)
