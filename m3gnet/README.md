# M3GNet
* M3GNet is an universal NNP, and is now contained in Materials Graph Library (MatGL): https://matgl.ai/

## Energy calculation
* Single point energy calculation can be done as follows.
* Model is built using ASE.

```python
import warnings
import matgl
from ase.build import add_adsorbate, fcc111, molecule
from matgl.ext.ase import PESCalculator

warnings.simplefilter("ignore")

# Create an FCC (111) surface model
slab = fcc111('Pt', size=(3, 3, 4), vacuum=10.0)

# Load molecule
mol = molecule("CO")

# Position the molecule above the surface
add_adsorbate(slab=slab, adsorbate=mol, height=2.5, position="fcc")

pot = matgl.load_model("M3GNet-MP-2021.2.8-PES")
slab.calc = PESCalculator(pot)

energy = slab.get_potential_energy()
print(f"Energy = {energy:5.3f} eV")
```

## Molecular dynamics
* Molecular dynamics (using ASE) can be done as follows.

```python
import warnings
import matgl
from ase import units
from ase.build import add_adsorbate, fcc111, molecule
from ase.constraints import FixAtoms
from ase.md import Langevin
from ase.visualize import view
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from matgl.ext.ase import PESCalculator

warnings.simplefilter("ignore")

# Create an FCC (111) surface model
slab = fcc111('Pt', size=(3, 3, 4), vacuum=10.0)

# Load molecule
mol = molecule("CO")

# Position the molecule above the surface
add_adsorbate(slab=slab, adsorbate=mol, height=2.5, position="fcc")

from ase.constraints import FixAtoms

# Fix the lower half of the slab
mask = [atom.tag >= 3 for atom in slab]
slab.set_constraint(FixAtoms(mask=mask))

pot = matgl.load_model("M3GNet-MP-2021.2.8-PES")
slab.calc = PESCalculator(pot)

# Define the MD simulation parameters
temperature_K = 300  # Kelvin
timestep = 1 * units.fs  # Time step in femtoseconds
friction = 0.10 / units.fs  # Friction coefficient for Langevin dynamics

MaxwellBoltzmannDistribution(slab, temperature_K=temperature_K)

# Initialize the Langevin dynamics
dyn = Langevin(slab, timestep=timestep, temperature_K=temperature_K, friction=friction, trajectory="md.traj")

# Run the MD simulation
dyn.run(500)
```

* Using pymatgen to build a model.

```python
from pymatgen.core import Lattice, Structure
from pymatgen.io.ase import AseAtomsAdaptor
import matgl
from matgl.ext.ase import PESCalculator
import warnings
from ase import units

warnings.simplefilter("ignore")

# Make structure with pymatgen
struct = Structure.from_spacegroup("Pm-3m", Lattice.cubic(4.5), ["Cs", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])

# Convert to ASE atoms
atoms = AseAtomsAdaptor.get_atoms(struct)
atoms *= [3, 3, 3]

pot = matgl.load_model("M3GNet-MP-2021.2.8-PES")
atoms.calc = PESCalculator(pot)

energy = atoms.get_potential_energy()

print(f"Energy = {energy:5.3f} eV")
```
