# CHGNet
* CHGNet is an universal NNP: https://github.com/CederGroupHub/chgnet
* Install: `pip install chgnet`

## Energy calculation
* Single point energy calculation can be done as follows.
* Model is built using ASE.

```python
from ase.build import bulk
from chgnet.model.dynamics import CHGNetCalculator
from chgnet.model.model import CHGNet
    
solid = bulk(name="Pt", crystalstructure="fcc", a=3.9)

chgnet = CHGNet.load()
potential = CHGNetCalculator(potential=chgnet, properties="energy")
solid.calc = potential
        
energy = solid.get_potential_energy()
print(f"energy = {energy:5.3} eV")
```

## Geometry optimization
```python
from ase.calculators.emt import EMT
from ase.build import fcc111
from ase.optimize import BFGS
from ase.constraints import FixAtoms
from chgnet.model.dynamics import CHGNetCalculator
from chgnet.model.model import CHGNet
from pymatgen.io.ase import AseAtomsAdaptor

surf = fcc111(symbol="Pt", size=[1, 1, 4], a=3.9, vacuum=12.0)
surf.pbc = True
c = FixAtoms(indices=[atom.index for atom in surf if atom.tag >= 3])
surf.constraints = c

chgnet = CHGNet.load()
potential = CHGNetCalculator(potential=chgnet, properties="energy")
surf.calc = potential
        
opt = BFGS(surf, trajectory="pt-relax.traj")
opt.run(fmax=0.05, steps=100)
```