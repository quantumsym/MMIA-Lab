#!/usr/bin/env python3
"""
2D Ising Model - Modular version with local bond energy function
Author: S.Magrì <s.magri@quantumsym.com>  luglio 2025
"""
from IsingModel2D import *

#------------------------------------------------------------------------
#
# Test parameters
L = 50      # lattice side length
p = 0.3      # probability of spin −1
J = 1.0      # coupling constant

# ------------------------------------------------------------------

if __name__ == "__main__":

    # Initialize lattice and analyze
    lattice = initialize_lattice(L, p)
    print_parameter(L, J, p)
    print_system_info(lattice, J, p)

    # Local right+bottom energy of a single site (example)
    i0, j0 = 5, 7
    e_rb = local_bond_energy_rb(lattice, i0, j0, J)
    print(f"Right+bottom energy of spin ({i0}, {j0}): {e_rb: .3f}\n")

    # set plot title
    plot_title = f"2D Ising Model {L}×{L} — p={p}, J={J}"
    # Visualize configuration
    plot_spin_configuration(lattice,title=plot_title)

