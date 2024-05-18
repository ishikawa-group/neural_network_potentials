from ase.io import read
from ase.visualize import view

optimized_atoms = read("test.traj", index=':')  # Read all frames of the track file
view(optimized_atoms)
