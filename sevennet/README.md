# SevenNet

- **SevenNet** (Scalable EquiVariance-Enabled Neural Network) is a graph neural network (GNN)-based interatomic potential package. It supports efficient atomistic simulations—including parallel molecular dynamics (MD) with LAMMPS—using pretrained models and a fine-tuning interface.
- GitHub Repository: [https://github.com/MDIL-SNU/SevenNet](https://github.com/MDIL-SNU/SevenNet)

## Installation

### Requirements

- **Python ≥ 3.8**
- **PyTorch ≥ 1.12.0** (recommended versions used internally include PyTorch 2.2.2 + CUDA 12.1.0, PyTorch 1.13.1 + CUDA 12.1.0, or PyTorch 1.12.0 + CUDA 11.6.2)

> **Important:** Install PyTorch manually according to your hardware before installing SevenNet.

### Installing SevenNet

To install via pip:

```bash
pip install sevenn
```

For the latest development version from GitHub:

```bash
pip install git+https://github.com/MDIL-SNU/SevenNet.git
```

Make sure to consult SevenNet GitHub Repository for new features and updates, as SevenNet is under active development.

## Usage

### Using the ASE Calculator

SevenNet provides an ASE interface through its calculator. Here are two common usage examples:

#### Basic ASE Calculator

Load a pretrained SevenNet model using a specified keyword and, if applicable, a modality (e.g., `mpa` or `omat24`):

```python
from sevenn.calculator import SevenNetCalculator

# Load a multi-fidelity model (using modal='mpa' or 'omat24' based on training DFT settings)
calc = SevenNetCalculator(model='7net-mf-ompa', modal='mpa')
```

#### Using CUDA-accelerated D3 Calculations

For dispersion-corrected calculations with GPU acceleration:

```python
from sevenn.calculator import SevenNetD3Calculator

# Replace '7net-0' with the desired model keyword and specify device
calc = SevenNetD3Calculator(model='7net-0', device='cuda')
```

_Tip:_ When the device is set to `'auto'`, SevenNet will use GPU acceleration if available.

### Command-line Interface for Training & Inference

SevenNet offers multiple CLI commands for managing preprocessing, training, inference, and model deployment:

1. **Input Generation**  
   Use `sevenn_preset` to generate an input YAML file:

   ```bash
   sevenn_preset base > input.yaml
   ```

   Preset keywords include `base`, `fine_tune`, `multi_modal`, `sevennet-0`, and `sevennet-l3i5`.

2. **Preprocessing** (Optional)  
   Preprocess your dataset with:

   ```bash
   sevenn_graph_build {dataset_path} {cutoff_radius}
   ```

   This creates preprocessed files (e.g., `sevenn_data/graph.pt`) that can speed up training and inference.

3. **Training**  
   With your `input.yaml` and preprocessed data ready, start training:

   ```bash
   sevenn input.yaml -s
   ```

   For multi-GPU training using PyTorch DDP:

   ```bash
   torchrun --standalone --nnodes {nodes} --nproc_per_node {GPUs} --no_python sevenn input.yaml -d
   ```

   _(Note: `batch_size` in `input.yaml` is per GPU.)_

4. **Inference**  
   Use the trained checkpoint to predict energies, forces, and stresses:

   ```bash
   sevenn_inference checkpoint_best.pth path_to_structures/*
   ```

   Results are saved in the `sevenn_infer_result` directory.

5. **Deployment for LAMMPS**  
   Deploy a model as a LAMMPS potential:
   ```bash
   sevenn_get_model 7net-0
   ```
   For parallel MD potentials:
   ```bash
   sevenn_get_model 7net-0 -p
   ```
   The resulting files/directories can be used with LAMMPS’ `e3gnn` pair_style.

## MD Simulation with LAMMPS

SevenNet supports MD simulations using LAMMPS:

- **Single-GPU MD:**  
  Use the serial model with the `e3gnn` pair*style.  
  \_Example Input:*

  ```
  units       metal
  atom_style  atomic
  pair_style  e3gnn
  pair_coeff  * * {path_to_serial_model} {chemical_species}
  ```

- **Multi-GPU MD:**  
  Use the parallel model with `e3gnn/parallel` pair*style.  
  \_Example Input:*
  ```
  units       metal
  atom_style  atomic
  pair_style  e3gnn/parallel
  pair_coeff  * * {num_message_layers} {directory_of_parallel_model} {chemical_species}
  ```
  _(Ensure one GPU per MPI process is available.)_

For detailed LAMMPS setup, refer to the provided scripts and documentation. A patch script (`sevenn_patch_lammps`) is available for compiling LAMMPS with SevenNet-specific modifications.

## Notebook Tutorials

Several Jupyter Notebook tutorials are available to help users:

- **From Scratch:** Learn to train SevenNet from scratch, perform predictions, structure relaxations, and generate EOS curves.
- **Fine-Tuning:** Fine-tune a pretrained model and compare its performance to the original model.

To access these notebooks, clone the repository:

```bash
git clone https://github.com/MDIL-SNU/sevennet_tutorial.git
```
