# ORB Models

- **ORB Models** provide pretrained neural network potentials for atomic simulations. They are designed for accurate energy, forces, and stress predictions and are optimized for large-scale simulations on MacOS and Linux.
- GitHub Repository: [https://github.com/orbital-materials/orb-models](https://github.com/orbital-materials/orb-models)

## Install

- Install the package via pip:
  ```bash
  pip install orb-models
  ```
- **Large-Scale Systems:** For simulations with ≳5k atoms (periodic) or ≳30k atoms (non-periodic), it is recommended to install cuML (requires CUDA). For example:

  ```bash
  # For CUDA versions >=11.4, <11.8:
  pip install --extra-index-url=https://pypi.nvidia.com "cuml-cu11==25.2.*"

  # For CUDA versions >=12.0, <13.0:
  pip install --extra-index-url=https://pypi.nvidia.com "cuml-cu12==25.2.*"
  ```

- Note: Orb Models are expected to run on MacOS and Linux. Windows support is not guaranteed.

## Usage

### Direct Usage

You can directly predict energies, forces, and stress using the pretrained models with the helper functions. For example:

```python
import ase
from ase.build import bulk
from orb_models.forcefield import atomic_system, pretrained
from orb_models.forcefield.base import batch_graphs

device = "cpu"  # Change to "cuda" if GPU is available
# Load a pretrained model (here using the orb-v3 conservative model with unlimited neighbors trained on OMAT24)
orbff = pretrained.orb_v3_conservative_inf_omat(
    device=device,
    precision="float32-high",  # Options: "float32-high", "float32-highest", or "float64"
)

# Create an ASE atoms object
atoms = bulk('Cu', 'fcc', a=3.58, cubic=True)

# Convert ASE atoms to an atom graph representation
graph = atomic_system.ase_atoms_to_atom_graphs(atoms, orbff.system_config, device=device)

# (Optional) Batch multiple graphs for faster inference
# graph = batch_graphs([graph, graph, ...])

result = orbff.predict(graph, split=False)

# Convert predictions back to an ASE atoms object
atoms = atomic_system.atom_graphs_to_ase_atoms(
    graph,
    energy=result["energy"],
    forces=result["forces"],
    stress=result["stress"]
)
```

### Usage with ASE Calculator

You can also integrate ORB Models with ASE by using the provided calculator:

```python
import ase
from ase.build import bulk
from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator

device = "cpu"  # or set to "cuda"
orbff = pretrained.orb_v3_conservative_inf_omat(
    device=device,
    precision="float32-high"
)
calc = ORBCalculator(orbff, device=device)

atoms = bulk('Cu', 'fcc', a=3.58, cubic=True)
atoms.calc = calc

# Compute the potential energy
print(f"Energy: {atoms.get_potential_energy()} eV")
```

This calculator can be used for geometry optimizations or MD simulations with any ASE-compatible code. For instance, to run a geometry optimization:

```python
from ase.optimize import BFGS

atoms.rattle(0.5)  # Perturb the atomic positions to leave the minimum-energy configuration
print("Rattled Energy:", atoms.get_potential_energy())

dyn = BFGS(atoms)
dyn.run(fmax=0.01)
print("Optimized Energy:", atoms.get_potential_energy())
```

## Floating Point Precision

ORB Models support three floating point precision types:

- `"float32-high"` (default for maximal acceleration with Nvidia A100/H100 GPUs)
- `"float32-highest"` (recommended for high-precision properties)
- `"float64"`  
  It is recommended to use `"float32-high"` unless your application requires higher precision.

## Finetuning

You can finetune a model on your own dataset, which should be formatted as an ASE SQLite database. Use the provided finetuning script:

```bash
python finetune.py --dataset=<dataset_name> --data_path=<your_data_path> --base_model=<base_model>
```

- **base_model** can be one of:
  - `"orb_v3_conservative_inf_omat"`
  - `"orb_v3_conservative_20_omat"`
  - `"orb_v3_direct_inf_omat"`
  - `"orb_v3_direct_20_omat"`
  - `"orb_v2"`

Finetuned model checkpoints will, by default, be saved in the `ckpts` folder. To load a finetuned model:

```python
from orb_models.forcefield import pretrained

model = getattr(pretrained, "<base_model>")(
    weights_path="<path_to_ckpt>",
    device="cpu",       # or "cuda"
    precision="float32-high"
)
```

**Caveats:**

- The finetuning script assumes the ASE database contains energy, forces, and stress. For molecular data without stress, manual modification is required.
- Early stopping is not implemented by default; you can use the command line argument `--save_every_x_epochs` for checkpointing and then retrospectively select the best model.
- The default learning rate schedule and training steps may need adjustment for your dataset.
