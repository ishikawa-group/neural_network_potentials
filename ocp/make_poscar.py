from ase import Atom, Atoms
from ase.io import read, write
from ase.build import sort

bulk = read("proton_diffusion/BaZrO3.cif")
replicate_size = 3
replicate = [replicate_size]*3
bulk = bulk*replicate
bulk = sort(bulk)
cell_length = bulk.cell.cellpar()
xpos = 0.50 * cell_length[0] / replicate_size
bulk.append(Atom("H", position=[xpos, 0, 0]))

write("POSCAR", bulk)
