# **DeePMD-kit**

- **DeePMD-kit** documentation: <https://deepmd.readthedocs.io>

## **1. Introduction**

DeePMD-kit is a machine learning-based tool that fits **first-principles** (DFT) potential energy surfaces (PES) to be used in **molecular dynamics (MD)** simulations. It provides **ab initio accuracy** at a fraction of the computational cost and integrates with **LAMMPS, GROMACS, OpenMM**, etc.

## **2. Workflow Overview**

1. **Prepare Data**: Convert AIMD/DFT data into DeePMD-kit format.
2. **Train**: Use `dp train input.json` to fit a model.
3. **Freeze**: Convert trained models into `.pb` files.
4. **Compress** _(Optional)_: Optimize `.pb` for efficiency.
5. **Test**: Validate energy/force predictions.
6. **Run MD**: Use the trained model in **LAMMPS** via `pair_style deepmd`.

---

## **3. Practical Guide**

### **3.1. Data Preparation**

DeePMD-kit uses a compressed NumPy-based format for efficient storage of atomic structures and properties. To convert raw simulation outputs into this format, we rely on the **dpdata** tool, which supports **VASP**, **CP2K**, **Gaussian**, **Quantum Espresso**, **ABACUS**, and **LAMMPS**.

Convert first-principles data to DeePMD-kit format:

```python
import dpdata, numpy as np

data = dpdata.LabeledSystem("00.data/abacus_md", fmt="abacus/md")
index_val = np.random.choice(len(data), 40, replace=False)
index_train = list(set(range(len(data))) - set(index_val))

data.sub_system(index_train).to_deepmd_npy("00.data/training_data")
data.sub_system(index_val).to_deepmd_npy("00.data/validation_data")
```

To convert other formats, change the `fmt` argument to one of:

- `"vasp/poscar"`
- `"cp2k/xyz"`
- `"gaussian/log"`
- `"qe/sssp"`
- `"lammps/dump"`

### **3.2. Training Configuration**

Define `input.json` for training:

```jsonc
{
  "model": {
    "type_map": ["H", "C"],
    "descriptor": { "type": "se_e2_a", "rcut": 6.0, "neuron": [25, 50, 100] },
    "fitting_net": { "neuron": [240, 240, 240] }
  },
  "training": {
    "training_data": { "systems": ["../00.data/training_data"] },
    "validation_data": { "systems": ["../00.data/validation_data"] },
    "numb_steps": 10000
  }
}
```

### **3.3. Train the Model**

Run:

```bash
dp train input.json
```

Monitor training loss in `lcurve.out`.

### **3.4. Freeze & Optimize Model**

Convert to a frozen model:

```bash
dp freeze -o graph.pb
```

(Optional) Compress the model:

```bash
dp compress -i graph.pb -o compress.pb
```

### **3.5. Model Testing**

Evaluate performance:

```bash
dp test -m graph.pb -s ../00.data/validation_data
```

For visualization:

```python
import dpdata, matplotlib.pyplot as plt

val_system = dpdata.LabeledSystem("../00.data/validation_data", fmt="deepmd/npy")
prediction = val_system.predict("graph.pb")

plt.scatter(val_system["energies"], prediction["energies"], alpha=0.5)
plt.xlabel("DFT Energy (eV)")
plt.ylabel("DP Predicted Energy (eV)")
plt.show()
```

### **3.6. Running MD with LAMMPS**

Write `in.lammps`:

```lammps
units metal
atom_style atomic
read_data conf.lmp
pair_style deepmd graph.pb
pair_coeff * *
timestep 0.001
run 5000
```

Run:

```bash
lmp -i in.lammps
```

---

## **4. Summary**

DeePMD-kit enables efficient **machine learning-based MD simulations** with **first-principles accuracy**. By leveraging **HPC**, it allows large-scale material and molecular modeling.
