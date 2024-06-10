from ase import Atom, Atoms
from ase.io import read, Trajectory
from ase.visualize import view
from ase.optimize import BFGS
from ase.calculators.emt import EMT
# from ase.md.langevin import Langevin
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units
from ase.constraints import FixAtoms
from ase.visualize.plot import plot_atoms
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import statsmodels.api as sm
from fairchem.core.models.model_registry import model_name_to_local_file
from fairchem.core.common.relaxation.ase_utils import OCPCalculator

# --- when using non-fine-tuned NNP
# model_name = "PaiNN-S2EF-OC20-All"
# checkpoint_path = model_name_to_local_file(model_name=model_name, local_cache="../checkpoints")

# --- load the fine-tuned NNP
checkpoint_path = "./checkpoints/2024-06-10-11-16-32/checkpoint.pt"

calc = OCPCalculator(checkpoint_path=checkpoint_path, cpu=False)

bulk = read("BaZrO3.cif")
replicate_size = 2
replicate = [replicate_size]*3
bulk = bulk*replicate
cell_length = bulk.cell.cellpar()
pos = 0.5 * cell_length[0] / replicate_size

# put hydrogen
bulk.append(Atom("H", position=[pos, 0, 0]))

bulk.set_calculator(calc)

traj_name = "test.traj"
temperature_K = 523
steps = 1000
timestep = 1.0*units.fs

MaxwellBoltzmannDistribution(bulk, temperature_K=temperature_K)
# dyn = Langevin(bulk, timestep=1.0*units.fs, temperature_K=temperature_K, friction=0.01/units.fs,
#                trajectory=traj_name)
dyn = NVTBerendsen(bulk, timestep=timestep, temperature_K=temperature_K, taut=0.5*1000*units.fs,
                   trajectory=traj_name)
dyn.run(steps=steps)

traj = read(traj_name, ":")

H_index = [i for i, x in enumerate(traj[0].get_chemical_symbols()) if x == "H"]

t0 = 0  # starting step
positions_all = np.array([traj[i].get_positions() for i in range(t0, len(traj))])

# position of H
positions = positions_all[:, H_index]
positions_x = positions[:, :, 0]
positions_y = positions[:, :, 1]
positions_z = positions[:, :, 2]

# total msd. sum along xyz axis & mean along Li atoms axis.
msd = np.mean(np.sum((positions-positions[0])**2, axis=2), axis=1)
real_timestep = timestep/units.fs/units.fs  # real (not ASE) femtosecond [10^-15 s]
print(real_timestep)
time = np.linspace(t0, len(msd), len(msd))/real_timestep

fontsize = 24
plt.plot(time, msd)
plt.xlabel("Time (ps)", fontsize=fontsize)
plt.ylabel("MSD (A^2)", fontsize=fontsize)
plt.tick_params(labelsize=fontsize)
plt.tight_layout()
plt.show()

subprocess.run(f"ase gui {traj_name}", shell=True)

model = sm.OLS(msd, time)
result = model.fit()
slope = result.params[0]
# slope, intercept, r_value, _, _ = stats.linregress(range(len(msd)), msd)
D = slope / 6   # divide by degree of freedom (x, y, z, -x, -y, -z)

plt.plot(time, msd, label="MSD")
plt.plot(time, time * slope, label="fitted line")
plt.xlabel("Time (ps)", fontsize=fontsize)
plt.ylabel("MSD (A^2)", fontsize=fontsize)
plt.tick_params(labelsize=fontsize)
plt.tight_layout()
plt.show()
print(f"Diffusion coefficient: {D*1e-16*1e12:6.4e} [cm^2/s]")
