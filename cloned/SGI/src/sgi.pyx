#!~/anaconda3/bin/python

# cython: language_level = 3 
# distutils: language = c++
# distutils: extra_compile_args = -std=c++11

import time
import numpy as np
import cython
from cython.parallel import prange, parallel
cimport openmp
from mpi4py import MPI
from cpython.mem cimport PyMem_Malloc, PyMem_Free


cdef extern from "<random>" namespace "std" nogil:
    cdef cppclass mt19937:
        #mt19937 generator.
        mt19937()
        mt19937(unsigned int seed)

    cdef cppclass uniform_int_distribution[T]:
        #Uniform integer distribution from [a,b] inclusive.
        uniform_int_distribution()
        uniform_int_distribution(T a, T b)
        T operator()(mt19937 gen)

    cdef cppclass uniform_real_distribution[T]:
        #Uniform real number distribution from [a,b) non-inclusive.
        uniform_real_distribution()
        uniform_real_distribution(T a, T b)
        T operator()(mt19937 gen)


cdef extern from "<cmath>" nogil:
    #exp function for use nogil multi-threading.
    double exp(double a)


@cython.boundscheck(False)
cdef double calc_hamiltonian(int[:,:] lattice, int[:,:] j_matrix, int threads):
    """Calculate Hamiltonian of lattice using j values in j_matrix.

    Arguments
    ---------
    lattice : 2D int array
        Lattice containing spins of +1 or -1
    j_matrix : 2D int array
        Array j values describing interactions between adjacent spins
    threads : int
        Number of threads for prange

    Returns
    -------
    H : double
        Hamiltonian
    """
    cdef:
        double H = 0.0
        int length = len(lattice)
        int i, j

    for i in prange(length, nogil=True, num_threads=threads):
        for j in range(length):
                #left, right, above, below
                #Periodic boundary conditions
                H += -(j_matrix[2*i+1,j]*lattice[i,j]*lattice[i,j-1]
                        +j_matrix[2*i+1,(j+1)%length]*lattice[i,j]*lattice[i,(j+1)%length]
                        +j_matrix[2*i,j]*lattice[i,j]*lattice[i-1,j]
                        +j_matrix[(2*i+2)%(2*length),j]*lattice[i,j]*lattice[(i+1)%length,j])
    H = H/2
    return H


@cython.boundscheck(False)
cdef calc_magnetisation(int[:,:] lattice, int threads):
    """Calculate magnetisation of lattice.

    Arguments
    ---------
    lattice : np.ndarray
        Lattice containing spins of +1 or -1
    threads : int
        Number of threads for prange
        
    Returns
    -------
    M : double
        Magnetisation of lattice
    """
    cdef:
        double M = 0.0
        int i,j
        int length = len(lattice)
    for i in prange(length, nogil=True, num_threads=threads):
        for j in range(length):
            M += lattice[i,j]
    M = M/(length**2)
    return M
    
    
"""
j_matrix relations
     l   l   l
  l  x l x l x
     l   l   l      
  l  x l x l x
     l   l   l
  l  x l x l x
"""


@cython.boundscheck(False)
cpdef main(int length, int steps, int rounds, double temperature, double ferro_prob, int threads):

    cdef:
        int nrounds = int(steps/rounds)  #No of snapshots
        int msteps  #No of attempted flips each thread attempts before each snapshot
        int[:,::1] lattice = np.zeros((length, length), dtype=np.int32)
        int[:,::1] j_matrix = np.zeros((length*2, length), dtype=np.int32)
        int[:,:,::1] lattice_list = np.zeros((nrounds+1, length, length), dtype=np.int32)
        int i, j, step, nround
        
        double kBT_inv
        double old_h = 0.0
        double new_h = 0.0
        double delta_h = 0.0
        double[:] energy_list = np.zeros(nrounds+1, dtype=float)
        double[:] magnetisation_list = np.zeros(nrounds+1, dtype=float)
        double init_time, final_time

        #OpenMP variables
        int thread
        int rows    #No of rows each thread in each process works on at any given time
        int offset
        int stencil = 2 #No of sections of lattice one thread works on in one stenciling cycle

        #RNG variables
        mt19937 * gen = <mt19937*> PyMem_Malloc(threads*sizeof(mt19937))    #Array of generators
        uniform_int_distribution[int] distr
        uniform_int_distribution[int] distc = uniform_int_distribution[int](0, length-1)
        uniform_real_distribution[double] distb = uniform_real_distribution[double](0, 1)

        #MPI specific values
        int MASTER = 0  #Master process
        int id, size    #ID of process, total number of processes
        int s
        int[:,::1] latticep #Part of lattice to receive using MPI_Gather
        int[:,::1] latticeg = np.zeros((length, length), dtype=np.int32)    #Buffer for MPI_Gather

    #Checks if PyMem_Calloc worked
    if not gen:
        raise MemoryError()

    comm = MPI.COMM_WORLD   #MPI Communicator
    id = comm.Get_rank()
    size = comm.Get_size()

    latticep = np.zeros((int(length/size), length), dtype=np.int32)
    rows = int(length/size/threads/stencil)
    msteps = int(rounds/stencil/size/threads)

    if rows*threads*size*stencil != length:
        print("LENGTH must be a multiple of no of processors, no of threads and %i!"%(
                                                stencil))
        return
    if steps%rounds != 0:
        print("ITERATION must be a multiple of ROUNDS!")
        print("ROUNDS must be a multiple of %s!"%(stencil))
        return
    if msteps*stencil*threads*size != rounds:
        print("ROUNDS must be a multiple of %s!"%(stencil))
        print("ROUNDS must be a multiple of no of processors, no of threads and %i\n"%stencil)
        return
    
    #kBT_inv will be bypassed if temperature==0
    if temperature != 0.0:
        kBT_inv = 1.0/(1.3806503e-23*temperature)

    #Unique generator for each thread in each process to prevent same random number
    for thread in range(threads):
        gen[thread] = mt19937(int(time.time())+id*threads+thread)

    if id == MASTER:
        lattice = np.random.choice((-1, 1), size=(length, length)).astype(np.int32)
        j_matrix = np.random.choice((-1, 1), size=(length*2, length),
                                    p=[1-ferro_prob, ferro_prob]
                                    ).astype(np.int32)

        #Snapshots of lattice
        lattice_list[0] = lattice
        energy_list[0] = calc_hamiltonian(lattice, j_matrix, threads)
        magnetisation_list[0] = calc_magnetisation(lattice, threads)

        #For timing purposes
        init_time = MPI.Wtime()
    
    #MASTER sends j_matrix to workers
    comm.Bcast(j_matrix, root=MASTER)

    for nround in range(nrounds):
        for offset in range(stencil):
            #MASTER sends updated lattice to workers
            comm.Bcast(lattice, root=MASTER)

            with nogil, parallel(num_threads=threads):
                thread = openmp.omp_get_thread_num()

                distr = uniform_int_distribution[int](rows*(stencil*id*threads+stencil*thread+offset),
                                                    rows*(stencil*id*threads+stencil*thread+offset+1)-1)

                for step in range(msteps):
                    #Choose random lattice site
                    i = distr(gen[thread])
                    j = distc(gen[thread])

                    #Calculate whether spin flip at (i,j) will reduce Hamiltonian
                    old_h = -(j_matrix[2*i+1,j]*lattice[i,j]*lattice[i,j-1]
                            +j_matrix[2*i+1,(j+1)%length]*lattice[i,j]*lattice[i,(j+1)%length]
                            +j_matrix[2*i,j]*lattice[i,j]*lattice[i-1,j]
                            +j_matrix[(2*i+2)%(2*length),j]*lattice[i,j]*lattice[(i+1)%length,j])
                    lattice[i,j] = -lattice[i,j]
                    new_h = -(j_matrix[2*i+1,j]*lattice[i,j]*lattice[i,j-1]
                            +j_matrix[2*i+1,(j+1)%length]*lattice[i,j]*lattice[i,(j+1)%length]
                            +j_matrix[2*i,j]*lattice[i,j]*lattice[i-1,j]
                            +j_matrix[(2*i+2)%(2*length),j]*lattice[i,j]*lattice[(i+1)%length,j])
                    delta_h = new_h-old_h

                    #Flip back spin if energy doesn't decreased
                    #or Boltzmann factor isn't fulfilled
                    if (not delta_h < 0):
                        #Hack to bypass infinite kBT_inv if temperature==0
                        if (temperature == 0.0 or
                                (not distb(gen[thread]) < exp(-delta_h*kBT_inv))):
                            lattice[i,j] = -lattice[i,j]
            
            #Master gathers part of lattice each worker was working on 
            latticep = lattice[rows*(stencil*id*threads):rows*(stencil*(id+1)*threads)]
            comm.Gather(latticep, latticeg, root=MASTER)

            #Rewrite lattice with gathered lattice parts
            if id == MASTER:
                lattice[:,:] = latticeg[:,:]

        #comm.Barrier()
            
        #Take snapshot every rounds of flips
        if id == MASTER:
            lattice_list[nround+1] = lattice
            energy_list[nround+1] = calc_hamiltonian(lattice, j_matrix, threads)
            magnetisation_list[nround+1] = calc_magnetisation(lattice, threads)
    
    if id == MASTER:
        final_time = MPI.Wtime()
        print("Elapsed time: %fs"%(final_time-init_time))

        #Save results
        np.savez_compressed('results', length=length,
                            iterations=steps, rounds=rounds,
                            temperature=temperature, ferro_prob=ferro_prob,
                            threads=threads, processes=size,
                            lattice_list=lattice_list,
                            energy_list=energy_list,
                            magnetisation_list=magnetisation_list,
                            j_matrix=j_matrix)

    PyMem_Free(gen)
    
    return
