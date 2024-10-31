from ase.build import surface, add_adsorbate, sort, molecule
from ase.io import read, write
from ase.cluster import Icosahedron
from ase.visualize import view

cluster = Icosahedron("Pd", noshells=4, latticeconstant=3.9)

bulk = read("BaZrO3.cif")
# bulk = read("MgO.cif")

vacuum = 16.0
surf = surface(lattice=bulk, indices=(1, 0, 0), layers=4, vacuum=vacuum)
surf *= [8, 8, 1]
surf.translate([0, 0, -vacuum+0.1])
surf.pbc = True

mol = molecule("H2")

# cluster + slab
add_adsorbate(slab=surf, adsorbate=cluster, height=9.0, offset=(0.5, 0.5))

# slab + molecule
add_adsorbate(slab=surf, adsorbate=mol, height=5.0, offset=(0.1, 0.1))
add_adsorbate(slab=surf, adsorbate=mol, height=5.0, offset=(0.9, 0.9))
add_adsorbate(slab=surf, adsorbate=mol, height=5.0, offset=(0.2, 0.2))
add_adsorbate(slab=surf, adsorbate=mol, height=5.0, offset=(0.8, 0.8))

add_adsorbate(slab=surf, adsorbate=mol, height=8.0, offset=(0.1, 0.1))
add_adsorbate(slab=surf, adsorbate=mol, height=8.0, offset=(0.9, 0.9))
add_adsorbate(slab=surf, adsorbate=mol, height=8.0, offset=(0.2, 0.2))
add_adsorbate(slab=surf, adsorbate=mol, height=8.0, offset=(0.8, 0.8))

surf = sort(surf)

write("POSCAR", surf)