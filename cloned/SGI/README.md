Spin Glass Ising
================

SGI is a Cython program for running 2D Ising Model Simulations
of Spin Glasses with Bimodal Frustration using the Edwards-Anderson Model.


Requirements
------------
- Python3
- time
- copy
- argparse
- numpy
- Cython
- openmp
- mpi4py
- cpython
- libcpp


Installation
------------
If you want to make all versions:
    
    make all

If you simply want the final product:
    
    make sgi


How to Run
----------

  |   Flag    |     | Description                                                           | Source            |
  | --------- | --- | --------------------------------------------------------------------- | ------------------|
  | --linearp | -lp | Run Python version without OpenMP or Open-MPI                         | sgi_linearp.py    |
  | --linear  | -l  | Run Cython verson without OpenMP or Open-MPI                          | sgi_linear.pyx    |
  | --linearc | -lc | Run Cython version with C function calls, without OpenMP or Open-MPI  | sgi_linearc.pyx   |
  | --openmp  | -t  | Run Cython version with C function calls and OpenMP                   | sgi_omp.pyx       |
  | --mpi     |-p   | Run Cython version with C function calls and Open-MPI                 | sgi_mpi.pyx       |
  | --all     |-a   | Run Cython version with C function calls, OpenMP and Open-MPI         | sgi.pyx           |

  | Argument        | Description                                                                           |
  | ---------       | ------------------------------------------------------------------------------------- |
  | matrix_length   | Length of 2D square matrix                                                            |
  | iterations      | Number of iterations (attempted flips, not Monte Carlo iteration) to make             |
  | rounds          | Takes a snapshot of the lattice every ROUNDS iterations                               |
  | temperature     | Temperature of system in Kelvin                                                       |
  | ferro_prob      | Probability of a coupling constant J_{ij} being +1; probability of -1 is then (1-p)   |
  | threads         | Number of threads for each process with OpenMP                                        |

Notes:
 - LENGTH must be a multiple of no of processors, no of threads and 2
 - ROUNDS must be a divisor or ITERATIONS
 - ROUNDS must be a multiple of no of processors, no of threads and 2


For example:

    mpiexec -n 2 python run_sgi.py -a 80 1000000 2000 0 0.7 4
    

runs SGI with 2 processes and 4 threads for a lattice of size 80x80, where about 0.7 of the bonds
are ferromagnetic and 0.3 of the bonds are anti-ferromagnetic, at 0K for 1,000,000 attempted spin flips,
taking a snapshot every 2000 attempted spin flips. The result will be stored in `results.npz', which can be plotted with:

    python plot_results.py results.npz
