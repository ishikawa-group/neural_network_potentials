# Using pre-trained models in ASE
* In this section, the usage of NNP(neural network potential) under ASE is shown.
* The corresponding Python script: `calculate.py`

1. See what pre-trained potentials are available

```python
from fairchem.core.models.model_registry import available_pretrained_models
print(available_pretrained_models)  # you can see the available models
```

2. Choose a checkpoint you want to use and download it automatically: e.g. GemNet-OC, trained on OC20 and OC22.

```python
from fairchem.core.models.model_registry import model_name_to_local_file
from fairchem.core.common.relaxation.ase_utils import OCPCalculator
from ase.build import fcc111, add_adsorbate
from ase.optimize import BFGS
import matplotlib.pyplot as plt
from ase.visualize import view

# checkpoint is downloaded by the following command
checkpoint_path = model_name_to_local_file("GemNet-OC-S2EFS-OC20+OC22", local_cache="./downloaded_checkpoints/")

# Define the model atomic system, a Pt(111) slab with an *O adsorbate!
slab = fcc111('Pt', size=(2, 2, 5), vacuum=10.0)
add_adsorbate(slab, 'O', height=1.2, position='fcc')

# Load the pre-trained checkpoint!
calc = OCPCalculator(checkpoint_path=checkpoint_path, cpu=False)
slab.set_calculator(calc)

# Run the optimization
opt = BFGS(slab, trajectory="test.traj")
opt.run(fmax=0.05, steps=100)

# Visualize the result
view(opt)
```

# Available Pretrained models
* Pretrained NNP models should be found in the following website:
	* https://fair-chem.github.io/core/model_checkpoints.html
