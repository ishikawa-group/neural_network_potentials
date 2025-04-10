# Neural Network Potentials

This repository summarizes popular neural network potentials (NNPs) that serve as efficient replacements for ab initio calculations.

> **Note:** This repository collects and summarizes the NNP models we have used. For the most up-to-date information, please refer to each model’s specific repository.

- **Webpage:** [https://ishikawa-group.github.io/neural_network_potentials/](https://ishikawa-group.github.io/neural_network_potentials/)

## List of NNP Models

1. **[M3GNet](./m3gnet/README.md)**
   - A versatile interatomic potential leveraging graph neural network architectures.
2. **[CHGNet](./chgnet/README.md)**
   - A universal neural network potential designed for accurate energy and force predictions.
3. **[MatterSim](./mattersim/README.md)**
   - A deep learning atomistic model for simulating materials across different elements, temperatures, and pressures.
4. **[SevenNet](./sevennet/README.md)**
   - A scalable, equivariance-enabled neural network for efficient parallel molecular dynamics simulations using LAMMPS.
5. **[MACE](./mace/README.md)**
   - A fast and accurate machine learning interatomic potential based on higher-order equivariant message passing.
6. **[ORB](./orb/README.md)**
   - Pretrained neural network potentials for atomic simulations, optimized for scalability and speed.
7. **[OpenCatalystProject (Fairchem)](./ocp/README.md)**
   - A framework focused on generating and using catalyst reaction potentials with NNPs.

## Other Model Training Approaches

**DeepMD-kit** is a package written in Python/C++, designed to minimize the effort required to build deep learning-based models of interatomic potential energy and force field and to perform molecular dynamics simulations.

- **[DeepMD-kit](./deepmd-kit/README.md)**

  - A toolkit for molecular dynamics simulations using deep neural network potentials.
