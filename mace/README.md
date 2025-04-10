# MACE

- **MACE** is a higher-order equivariant message passing neural network designed for building fast and accurate machine learning interatomic potentials.
- GitHub Repository: [https://github.com/ACEsuit/mace](https://github.com/ACEsuit/mace)

## Install

- To install the recommended version (based on PyTorch):
  ```bash
  pip install mace-torch
  ```
- If you prefer to install from source or need more detailed installation instructions, please refer to the official documentation: [MACE Documentation](https://mace-docs.readthedocs.io)
- Note: Ensure your Python version is ≥ 3.7 (for openMM, Python 3.9 is recommended) and that your PyTorch version meets the requirements.

## Training

- Train a MACE model using the `mace_run_train` script. Here is an example:
  ```bash
  mace_run_train \
      --name="MACE_model" \
      --train_file="train.xyz" \
      --valid_fraction=0.05 \
      --test_file="test.xyz" \
      --config_type_weights='{"Default":1.0}' \
      --E0s='{1:-13.663181292231226, 6:-1029.2809654211628, 7:-1484.1187695035828, 8:-2042.0330099956639}' \
      --model="MACE" \
      --hidden_irreps='128x0e + 128x1o' \
      --r_max=5.0 \
      --batch_size=10 \
      --max_num_epochs=1500 \
      --stage_two \
      --start_stage_two=1200 \
      --ema \
      --ema_decay=0.99 \
      --amsgrad \
      --restart_latest \
      --device=cuda
  ```
- Alternatively, you can place the parameters in a YAML file and specify it on the command line with `--config="your_configs.yaml"` (command line arguments override the YAML settings).

## Evaluation

- Evaluate the trained MACE model (to compute properties on an XYZ file) using the `mace_eval_configs` script:
  ```bash
  mace_eval_configs \
      --configs="your_configs.xyz" \
      --model="your_model.model" \
      --output="./your_output.xyz"
  ```

## Example Usage in ASE

### MACE-MP: Materials Project Potentials

- MACE-MP provides a universal potential trained on Materials Project datasets.
- Example code:

  ```python
  from mace.calculators import mace_mp
  from ase import build

  # Build a water molecule
  atoms = build.molecule('H2O')

  # Initialize the MACE-MP calculator (using the "medium" model and disabling dispersion corrections)
  calc = mace_mp(model="medium", dispersion=False, default_dtype="float32", device='cuda')
  atoms.calc = calc

  energy = atoms.get_potential_energy()
  print(f"energy = {energy:5.3f} eV")
  ```

### MACE-OFF: Transferable Organic Potentials

- MACE-OFF is designed for organic chemistry systems such as molecules, crystals, and molecular liquids.
- Example code:

  ```python
  from mace.calculators import mace_off
  from ase import build

  # Build a water molecule
  atoms = build.molecule('H2O')

  # Initialize the MACE-OFF calculator (using the "medium" model)
  calc = mace_off(model="medium", device='cuda')
  atoms.calc = calc

  energy = atoms.get_potential_energy()
  print(f"energy = {energy:5.3f} eV")
  ```

## Pretrained Foundation Models

- MACE offers several pretrained foundation models targeting different domains:
  - **MACE-MP series** (e.g., MACE-MP-0a, MACE-MP-0b3, MACE-MPA-0): for materials science.
  - **MACE-OFF series**: for organic chemistry.
- For the latest releases, please refer to their respective GitHub Release pages:
  - MACE-MP: [https://github.com/ACEsuit/mace-mp](https://github.com/ACEsuit/mace-mp)
  - MACE-OFF: [https://github.com/ACEsuit/mace-off](https://github.com/ACEsuit/mace-off)

## Advanced Topics

- **Fine-tuning Foundation Models**  
  To fine-tune an existing foundation model, specify the additional parameter `--foundation_model` during training. For example:
  ```bash
  mace_run_train \
      --name="MACE" \
      --foundation_model="small" \
      --train_file="train.xyz" \
      --valid_fraction=0.05 \
      --test_file="test.xyz" \
      --energy_weight=1.0 \
      --forces_weight=1.0 \
      --E0s="average" \
      --lr=0.01 \
      --scaling="rms_forces_scaling" \
      --batch_size=2 \
      --max_num_epochs=6 \
      --ema \
      --ema_decay=0.99 \
      --amsgrad \
      --default_dtype="float32" \
      --device=cuda \
      --seed=3
  ```
- **CUDA Acceleration**  
  MACE can leverage the cuEquivariance library for CUDA acceleration. For more details, see [CUDA Acceleration](https://mace-docs.readthedocs.io/en/latest/guide/cuda_acceleration.html)
- **Multi-GPU Training**  
  In multi-GPU environments, use the `--distributed` flag to employ PyTorch's DistributedDataParallel module.
- **Online Data Loading for Large Datasets**  
  For very large datasets, it is recommended to preprocess your data using `preprocess_data.py` and then load data online during training.

## Caching & Development

- **Caching**
  - By default, automatically downloaded models are saved in `~/.cache/mace`. You can change the cache path by setting the environment variable `XDG_CACHE_HOME`.
- **Development and Code Checking**
  - It is recommended to install the development dependencies:
    ```bash
    pip install -e ".[dev]"
    pre-commit install
    ```
  - This project uses tools like black, isort, pylint, and mypy for formatting and static analysis.

---

This document is designed to help newcomers quickly understand and start using MACE. For more detailed information and parameter descriptions, please refer to the [MACE Documentation](https://mace-docs.readthedocs.io) and the README file in the GitHub repository.
