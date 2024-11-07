from pymatgen.core import Lattice, Structure
from pymatgen.io.ase import AseAtomsAdaptor
import matgl
from matgl.ext.ase import PESCalculator
import warnings

from ase import units
from ase.io import read
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
import os

warnings.simplefilter("ignore")

trajfile = "md.traj"
logfile = "md.log"

os.system(f"rm {trajfile} {logfile}")

pot = matgl.load_model("M3GNet-MP-2021.2.8-PES")
calc = PESCalculator(pot)
mol = read("POSCAR")
mol.calc = calc

MaxwellBoltzmannDistribution(mol, temperature_K=500)
# dyn = NVTBerendsen(mol, 2*units.fs, 500, taut=20*units.fs, trajectory=trajfile, logfile=logfile)
dyn = Langevin(atoms=mol, timestep=2.0*units.fs, temperature_K=500,
               trajectory=trajfile, logfile=logfile, friction=0.1/units.fs, loginterval=10)
dyn.run(steps=1000)
print("Done")
