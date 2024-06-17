import os 
# os.environ['OPENBLAS_NUM_THREADS'] = '1'  # if openblas error happens, uncomment this
import math
import subprocess
import matplotlib.pyplot as plt
import statsmodels.api as sm
import argparse
import numpy as np
from fairchem.core.models.model_registry import model_name_to_local_file
from ase import Atom, Atoms
from ase.io import read, write, Trajectory
from ase.visualize import view
from ase.optimize import BFGS
from ase.calculators.emt import EMT
from ase.md.langevin import Langevin
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units
from ase.constraints import FixAtoms
from ase.visualize.plot import plot_atoms
from fairchem.core.common.relaxation.ase_utils import OCPCalculator

# --- when using non-fine-tuned NNP
# model_name = "PaiNN-S2EF-OC20-All"
# checkpoint_path = model_name_to_local_file(model_name=model_name, local_cache="../checkpoints")

parser = argparse.ArgumentParser()
parser.add_argument("--maxtime_ps", default=1)
parser.add_argument("--show_plot", action="store_true")
args = parser.parse_args()

maxtime_ps = float(args.maxtime_ps)
show_plot = args.show_plot

cpline  = subprocess.check_output(["grep", "checkpoint_dir", "train.txt"])
cpdir   = cpline.decode().strip().replace(" ", "").split(":")[-1]
checkpoint_path = cpdir + "/checkpoint.pt"

calc = OCPCalculator(checkpoint_path=checkpoint_path, cpu=False)

bulk = read("BaZrO3.cif")
###
replicate_size = 4
each = 10  # step to save trajectory
###
replicate = [replicate_size]*3
bulk = bulk*replicate
cell_length = bulk.cell.cellpar()
pos = 0.5 * cell_length[0] / replicate_size

# put hydrogen
bulk.append(Atom("H", position=[  pos,     0,     0]))
bulk.append(Atom("H", position=[3*pos, 2*pos,     0]))
bulk.append(Atom("H", position=[    0, 3*pos, 2*pos]))
bulk.append(Atom("H", position=[5*pos,     0,     0]))
bulk.append(Atom("H", position=[8*pos, 7*pos,     0]))
bulk.append(Atom("H", position=[    0, 8*pos, 7*pos]))

bulk.calc = calc

traj_name = "test.traj"
temperature_K = 1000
timestep = 1.0*units.fs  # in fs. Use 1 or 0.5 fs.
t0 = 0.1  # starting time for taking MSD [ps].
loginterval = 10

maxtime_ps = maxtime_ps + t0  # extend maxtime_ps as we discard the initial t0 ps

steps   = math.ceil(maxtime_ps/(timestep*1e-3))
t0steps = math.ceil(t0/(timestep*1e-3))

print(f"Temperature [K]: {temperature_K}", flush=True)
print(f"Maximum time [ps]: {maxtime_ps} (discard initial {t0} [ps])", flush=True)
print(f"Number of steps: {steps} (calculate), {int(steps/loginterval)} (write to trajectory)", flush=True)

MaxwellBoltzmannDistribution(bulk, temperature_K=temperature_K)
dyn = Langevin(bulk, timestep=timestep, temperature_K=temperature_K, friction=0.01/units.fs,
               trajectory="tmp.traj", logfile="md.log", loginterval=loginterval)  # friction: 0.01-0.1
# dyn = NVTBerendsen(bulk, timestep=timestep, temperature_K=temperature_K, taut=0.01*(1000*units.fs),
#                   trajectory="tmp.traj", logfile="md.log", loginterval=loginterval)  # tout: 0.01-0.1
dyn.run(steps=steps)

# saveing trajectory file with some interval, as the file becomes too big
tmptraj = read("tmp.traj", f"::{each}")
write(traj_name, tmptraj)

# load trajectory
traj = read("tmp.traj", ":")

H_index = [i for i, x in enumerate(traj[0].get_chemical_symbols()) if x == "H"]

# position of all atoms
start_pos = math.ceil(t0steps/loginterval)  # where to start data
all_pos   = math.ceil(steps/loginterval)    # should be same with len(traj)
positions_all = np.array([traj[i].get_positions() for i in range(start_pos, all_pos)])

# position of H
positions = positions_all[:, H_index]

# total msd. sum along xyz axis & mean along Li atoms axis.
msd  = np.mean(np.sum((positions-positions[start_pos])**2, axis=2), axis=1)
time = np.linspace(0, maxtime_ps, len(msd))

model = sm.OLS(msd, time)
result = model.fit()
slope = result.params[0]
D = slope / 6   # divide by degree of freedom (x, y, z, -x, -y, -z)
print(f"Diffusion coefficient: {D*1e-16*1e12:6.4e} [cm^2/s]")

fontsize = 24
plt.plot(time, msd, label="MSD")
plt.plot(time, time * slope, label="fitted line")
plt.xlabel("Time (ps)", fontsize=fontsize)
plt.ylabel("MSD (A^2)", fontsize=fontsize)
plt.tick_params(labelsize=fontsize)
plt.tight_layout()
plt.savefig("result.png")
if show_plot:
    plt.show()

subprocess.run("rm tmp.traj md.log", shell=True)

