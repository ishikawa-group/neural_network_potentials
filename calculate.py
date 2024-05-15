from fairchem.core.models.model_registry import model_name_to_local_file
from fairchem.core.common.relaxation.ase_utils import OCPCalculator
from ase.build import fcc111, add_adsorbate
from ase.optimize import BFGS
import matplotlib.pyplot as plt
from ase.visualize import view


# checkpoint is downloaded by the following command
checkpoint_path = model_name_to_local_file("GemNet-OC-S2EFS-OC20+OC22",
                                            local_cache="./tmp/ocp_checkpoints/")

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
view("test.traj")
