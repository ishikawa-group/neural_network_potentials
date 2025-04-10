# MatterSim

- **MatterSim** is a deep learning atomistic model capable of simulating materials across a range of elements, temperatures, and pressures.
- GitHub Repository: [https://github.com/microsoft/mattersim](https://github.com/microsoft/mattersim)

## Installation

### Prerequisites

- **Python ≥ 3.9** is required (although MatterSim works with later versions, for optimal compatibility Python 3.9 is recommended).

### Conda Environment (Recommended)

For a clean installation, create and activate a conda environment:

```bash
conda create -n mattersim python=3.9
conda activate mattersim
```

### Install from PyPI

Install MatterSim using pip:

```bash
pip install mattersim
```

_Tip:_ Downloading dependencies may take some time.

To install the latest development version directly from GitHub:

```bash
pip install git+https://github.com/microsoft/mattersim.git
```

### Install from Source

Clone the repository and install using mamba (recommended for faster dependency resolution):

```bash
git clone git@github.com:microsoft/mattersim.git
cd mattersim
mamba env create -f environment.yaml
mamba activate mattersim
pip install -e .
```

## Pretrained Models

MatterSim offers two pre-trained MatterSim-v1 models (based on the M3GNet architecture) located in the `pretrained_models` folder:

- **MatterSim-v1.0.0-1M:** A mini version that is faster to run.
- **MatterSim-v1.0.0-5M:** A larger version that provides more accuracy.

By default, the 1M version is loaded. To switch to the 5M version, set the `load_path` parameter in the MatterSim calculator accordingly.

## Usage

### Basic Example

The following minimal code demonstrates how to perform a simple calculation with MatterSim:

```python
import torch
from loguru import logger
from ase.build import bulk
from ase.units import GPa
from mattersim.forcefield import MatterSimCalculator

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Running MatterSim on {device}")

# Create an ASE atoms object for silicon in diamond structure
si = bulk("Si", "diamond", a=5.43)
si.calc = MatterSimCalculator(device=device)

logger.info(f"Energy (eV)                 = {si.get_potential_energy()}")
logger.info(f"Energy per atom (eV/atom)   = {si.get_potential_energy()/len(si)}")
logger.info(f"Forces of first atom (eV/A) = {si.get_forces()[0]}")
logger.info(f"Stress[0][0] (eV/A^3)       = {si.get_stress(voigt=False)[0][0]}")
logger.info(f"Stress[0][0] (GPa)          = {si.get_stress(voigt=False)[0][0] / GPa}")
```

### Switching Pretrained Models

To load the MatterSim-v1.0.0-5M checkpoint, update the calculator as follows:

```python
from mattersim.forcefield import MatterSimCalculator

# Provide the alternate load path for the 5M version
calc = MatterSimCalculator(load_path="MatterSim-v1.0.0-5M.pth", device=device)
```

### Note for macOS Users

If you’re on macOS with Apple Silicon, numerical instability with the MPS backend may occur. It is recommended to run MatterSim on the CPU on these systems.

## Finetuning

MatterSim provides a finetuning script that allows you to refine the pre-trained model on your custom dataset (formatted as an ASE SQLite database). For example:

```bash
torchrun --nproc_per_node=1 src/mattersim/training/finetune_mattersim.py \
    --load_model_path mattersim-v1.0.0-1m \
    --train_data_path tests/data/high_level_water.xyz
```

For more details on finetuning, please refer to the MatterSim documentation.
