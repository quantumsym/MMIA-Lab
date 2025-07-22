#!~/anaconda3/bin/python

import sys
import argparse
from mpi4py import MPI

from src import sgi_linearp
from lib import sgi_linear
from lib import sgi_linearc
from lib import sgi_omp
from lib import sgi_mpi
from lib import sgi


parser = argparse.ArgumentParser(description="2D Ising Model Simulation of Spin Glasses "
                                                "using Bimodal Frustration in the Edwards-Anderson Model")

multi = parser.add_mutually_exclusive_group(required=True)
multi.add_argument('--linearp', '-lp', action='store_true',
                    help="Run Python version without OpenMP or Open-MPI")
multi.add_argument('--linear', '-l', action='store_true',
                    help="Run Cython verson without OpenMP or Open-MPI")
multi.add_argument('--linearc', '-lc', action='store_true',
                    help="Run Cython version with C function calls, without OpenMP or Open-MPI")
multi.add_argument('--openmp', '-t', action='store_true',
                    help="Run Cython version with C function calls and OpenMP")
multi.add_argument('--mpi', '-p', action='store_true',
                    help="Run Cython version with C function calls and Open-MPI (requires mpiexec or mpirun)")
multi.add_argument('--all', '-a', action='store_true',
                    help="Run Cython version with C function calls, OpenMP and Open-MPI (requires mpiexec or mpirun)")

#Used to determine whether THREADS is required
runtype = parser.parse_known_args()[0]

parser.add_argument('matrix_length', type=int,
                    help="Length of 2D square matrix")
parser.add_argument('iterations', type=int,
                    help="Number of iterations (attempted flips, not Monte Carlo iteration) to make")
parser.add_argument('rounds', type=int,
                    help="Takes a snapshot of the lattice every ROUNDS iterations")
parser.add_argument('temperature', type=float,
                    help="Temperature of system in Kelvin")
parser.add_argument('ferro_prob', type=float,
                    help="Probability of a coupling constant J_{ij} being +1; probability of -1 is then (1-p)")
if runtype.all or runtype.openmp:
    parser.add_argument('threads', type=int,
                        help="Number of threads for each process with OpenMP")
args = parser.parse_args()

#Rules for arguments, further runtype specific arguments in individual src files
if args.matrix_length <= 0:
    sys.exit("LENGTH must be > 0!")
elif args.matrix_length <= 0:
    sys.exit("ITERATIONS must be > 0!")
elif args.rounds <= 0:
    sys.exit("ROUNDS must be > 0!")
elif args.iterations%args.rounds != 0:
    sys.exit("ROUNDS must be a divisor of ITERATIONS!")
elif not (args.ferro_prob >= 0 and args.ferro_prob <= 1):
    sys.exit("FERRO_PROB must be between 0 and 1 inclusive!")

if args.all:
    sgi.main(args.matrix_length, args.iterations, args.rounds, args.temperature, args.ferro_prob, args.threads)
elif args.mpi:
    sgi_mpi.main(args.matrix_length, args.iterations, args.rounds, args.temperature, args.ferro_prob)
elif args.openmp:
    sgi_omp.main(args.matrix_length, args.iterations, args.rounds, args.temperature, args.ferro_prob, args.threads)
elif args.linearc:
    sgi_linearc.main(args.matrix_length, args.iterations, args.rounds, args.temperature, args.ferro_prob)
elif args.linear:
    sgi_linear.main(args.matrix_length, args.iterations, args.rounds, args.temperature, args.ferro_prob)
elif args.linearp:
    sgi_linearp.main(args.matrix_length, args.iterations, args.rounds, args.temperature, args.ferro_prob)

sys.exit()

