# CHGNet
* CHGNet is an universal NNP: https://github.com/CederGroupHub/chgnet

## Install 
* `pip install chgnet`
* New version of CHGNet (0.4.0 as of 2025/2/22) requires `torch==2.4.0` or newer. If you use M3GNet in the same environment, this leads the conflict of PyTorch version so one should keep `chgnet==0.3.8`.

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
from ase.build import fcc111
from ase.optimize import FIRE
from ase.constraints import FixAtoms
from chgnet.model.dynamics import CHGNetCalculator
from chgnet.model.model import CHGNet

surf = fcc111(symbol="Pt", size=[1, 1, 4], a=3.9, vacuum=12.0)
surf.pbc = True
c = FixAtoms(indices=[atom.index for atom in surf if atom.tag >= 3])
surf.constraints = c

chgnet = CHGNet.load()
potential = CHGNetCalculator(potential=chgnet, properties="energy")
surf.calc = potential
        
opt = FIRE(surf, trajectory="pt-relax.traj")
opt.run(fmax=0.05, steps=100)
```