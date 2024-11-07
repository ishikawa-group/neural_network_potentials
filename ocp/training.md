# Training and evaluating custom models on OCP datasets
## Getting Started
* This section is a tutorial for training and evaluating models for each S2EF.
* `main.py` serves as the entry point to run any task. This script requires two command line arguments at a minimum:
	1. `--mode MODE`: MODE can be train, predict or run-relaxations 
		to train a model, make predictions using an existing model, or run relaxations using an existing model, respectively.
	2. `--config-yml PATH`: PATH is the path to a YAML configuration file.
        The configs directory contains a number of example config files.
* Running `main.py` directly runs the model on a single CPU or GPU:
  * `python main.py --mode train --config-yml configs/TASK/SIZE/MODEL/MODEL.yml`
* The test case is already prepared: `python main.py --mode train --config-yml ./schnet.yml`
* Training results are stored in the log directory `logs/tensorboard/[TIMESTAMP]` where `[TIMESTAMP]` is starting time.
* You can see the training result by `tensorboard --logdir logs/tensorboard/[TIMESTAMP]`

### OC20
#### Structure to Energy and Forces (S2EF)
* In the S2EF task, the model takes the positions of the atoms as input and predicts the adsorption energy and per-atom forces.
* To train a model for the S2EF, you can use the OCPTrainer and TrajectoryLmdb dataset by specifying your configuration file:

```yaml
trainer: ocp

dataset:
  # Training data
  train:
    src: [Path to training data]
    normalize_labels: True
    # Mean and standard deviation of energies
    target_mean: -0.7586356401443481
    target_std: 2.981738567352295
    # Mean and standard deviation of forces
    grad_target_mean: 0.0
    grad_target_std: 2.981738567352295
  # Val data (optional)
  val:
    src: [Path to validation data]
  # Test data (optional)
  test:
    src: [Path to test data]
```

* You can find examples configuration files in `configs/s2ef`
* The checkpoint is stored in `checkpoints/[TIMESTAMP]/checkpoint.pt`
* The "checkpoint.pt" is the checkpoint at the last step, while "best_checkpoint.pt" is that for smallest validation error.
  Thus it is better to use "best_checkpoint.pt" in principle.
* Next, run this model on the test data: `python main.py --mode predict --config-yml configs/s2ef/2M/schnet/schnet.yml \
        --checkpoint checkpoints/[TIMESTAMP]/checkpoint.pt`
* The predictions are stored in `[RESULTS_DIR]/ocp_predictions.npz`

### Training OC20 models with total energies (S2EF)
* To train/validate an OC20 S2EF model on total energies instead of adsorption energies, you need to change the config file.
* They include setting as follows:
  + `dataset: oc22_lmdb`
  + `prediction_dtype: float32`
  + `train_on_oc20_total_energies: True`
  + `oc20_ref: path/to/oc20_ref.pkl`
* Also, please note that our evaluation server does not currently support OC20 total energy models.

```yaml
task:
  prediction_dtype: float32
  # ...

dataset:
  format: oc22_lmdb
  train:
    src: data/oc20/s2ef/train
    normalize_labels: False
    train_on_oc20_total_energies: True
    # download at https://dl.fbaipublicfiles.com/opencatalystproject/data/oc22/oc20_ref.pkl
    oc20_ref: path/to/oc20_ref.pkl
  val:
    src: data/oc20/s2ef/val_id
    train_on_oc20_total_energies: True
    oc20_ref: path/to/oc20_ref.pkl
```

## OC22
### Structure to Total Energy and Forces (S2EF-Total)
* The S2EF-Total task takes a structure and predicts the total DFT energy and per-atom forces.
* This differs from the original OC20 S2EF task because it predicts total energy instead of adsorption energy.
* To train an OC22 S2EF-Total model, you need the OC22LmdbDataset by including these lines in your configuration file:

```yaml
dataset:
  format: oc22_lmdb # Use the OC22LmdbDataset
  ...
```

* You can find examples configuration files in `configs/oc22/s2ef`.

## Available pre-trained models
### OC20
* SchNet-S2EF-OC20-200k, SchNet-S2EF-OC20-2M, SchNet-S2EF-OC20-20M, SchNet-S2EF-OC20-All
* SpinConv-S2EF-OC20-2M, SpinConv-S2EF-OC20-All
* GemNet-dT-S2EF-OC20-2M, GemNet-dT-S2EF-OC20-All
* PaiNN-S2EF-OC20-All
* GemNet-OC-S2EF-OC20-2M, GemNet-OC-S2EF-OC20-All, GemNet-OC-S2EF-OC20-All+MD, GemNet-OC-Large-S2EF-OC20-All+MD
* SCN-S2EF-OC20-2M, SCN-t4-b2-S2EF-OC20-2M, SCN-S2EF-OC20-All+MD
* eSCN-L4-M2-Lay12-S2EF-OC20-2M, eSCN-L6-M2-Lay12-S2EF-OC20-2M, eSCN-L6-M2-Lay12-S2EF-OC20-All+MD, 
  eSCN-L6-M3-Lay20-S2EF-OC20-All+MD
* EquiformerV2-83M-S2EF-OC20-2M, EquiformerV2-31M-S2EF-OC20-All+MD, EquiformerV2-153M-S2EF-OC20-All+MD
* SchNet-S2EF-force-only-OC20-All
* DimeNet++-force-only-OC20-All, DimeNet++-Large-S2EF-force-only-OC20-All, DimeNet++-S2EF-force-only-OC20-20M+Rattled, 
  DimeNet++-S2EF-force-only-OC20-20M+MD
* CGCNN-IS2RE-OC20-10k, CGCNN-IS2RE-OC20-100k, CGCNN-IS2RE-OC20-All
* DimeNet-IS2RE-OC20-10k, DimeNet-IS2RE-OC20-100k, DimeNet-IS2RE-OC20-all
* SchNet-IS2RE-OC20-10k, SchNet-IS2RE-OC20-100k, SchNet-IS2RE-OC20-All
* DimeNet++-IS2RE-OC20-10k, DimeNet++-IS2RE-OC20-100k, DimeNet++-IS2RE-OC20-All
* PaiNN-IS2RE-OC20-All

### OC22
* GemNet-dT-S2EFS-OC22, GemNet-OC-S2EFS-OC22, GemNet-OC-S2EFS-OC20+OC22, GemNet-OC-S2EFS-nsn-OC20+OC22, 
  GemNet-OC-S2EFS-OC20->OC22 
* EquiformerV2-lE4-lF100-S2EFS-OC22

### ODAC
* SchNet-S2EF-ODAC
* DimeNet++-S2EF-ODAC
* PaiNN-S2EF-ODAC
* GemNet-OC-S2EF-ODAC
* eSCN-S2EF-ODAC
* EquiformerV2-S2EF-ODAC, EquiformerV2-Large-S2EF-ODAC
* Gemnet-OC-IS2RE-ODAC
* eSCN-IS2RE-ODAC
* EquiformerV2-IS2RE-ODAC
