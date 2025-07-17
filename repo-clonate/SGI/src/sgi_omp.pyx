#!~/anaconda3/bin/python

# cython: language_level = 3
# distutils: language = c++
# distutils: extra_compile_args = -std=c++11

import time
import numpy as np
import cython
from cython.parallel import prange, parallel
cimport openmp
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from libcpp.vector cimport vector


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
    
    cdef cppclass discrete_distribution[T]:
        #Discrete distribution for [0, len(vector)] with probabilities given in vector
        discrete_distribution()
        discrete_distribution(vector.iterator first, vector.iterator last)
        T operator()(mt19937 gen)
        

cdef extern from "<cmath>" nogil:
    #exp function for use nogil multi-threading.
    double exp(double a)


@cython.boundscheck(False)
cdef init_lattice_2d(int lengthx, int lengthy, double upprob, mt19937 * gen, int threads):
    """Initialise 2D spin lattice.

    threads : int
        Number of threads for prange
    """
    cdef:
        int i, j
        int thread
        int[:,::1] lattice = np.zeros((lengthx, lengthy), dtype=np.int32)
        vector[double] spin_prob = [1-upprob, 0.0, upprob]  #Probability of [0,1,2]
        #Gives pointers of beginning and end as arguments
        discrete_distribution[int] disti = discrete_distribution[int](spin_prob.begin(), spin_prob.end())
        
    for i in prange(lengthx, nogil=True, num_threads=threads):
        thread = openmp.omp_get_thread_num()
        for j in range(lengthy):
            lattice[i,j] = disti(gen[thread])-1
    return lattice


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
        double H = 0
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
cdef double calc_magnetisation(int[:,:] lattice, int threads):
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
        int nrounds = int(steps/rounds) #No of snapshots
        int msteps   #No of attempted flips each thread attempts before each snapshot
        int[:,::1] lattice
        int[:,::1] j_matrix
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
        int thread  #Thread no
        int rows    #No of rows each thread works on at any given time
        int offset
        int stencil = 2 #No of sections of lattice one thread works on in one stenciling cycle

        #RNG variables
        mt19937 * gen = <mt19937*> PyMem_Malloc(threads*sizeof(mt19937))    #Array of generators
        uniform_int_distribution[int] distr
        uniform_int_distribution[int] distc = uniform_int_distribution[int](0, length-1)
        uniform_real_distribution[double] distb = uniform_real_distribution[double](0, 1)

    #Checks if PyMem_Calloc worked
    if not gen:
        raise MemoryError()
    
    rows = int(length/threads/stencil)
    msteps = int(rounds/stencil/threads)
    
    if rows*stencil*threads != length:
        print("LENGTH must be a multiple of no of threads and %i\n"%stencil)
        return
    if msteps*stencil*threads != rounds:
        print("ROUNDS must be >= 2")
        print("ROUNDS must be a multiple of no of threads and %i\n"%stencil)
        return

    #kBT_inv will be bypassed if temperature==0
    if temperature != 0.0:
        kBT_inv = 1.0/(1.3806503e-23*temperature)

    #Unique generator for each thread to prevent same random number
    for thread in range(threads):
        gen[thread] = mt19937(int(time.time())+thread)
    
    lattice = init_lattice_2d(length, length, 0.5, gen, threads)
    j_matrix = init_lattice_2d(length*2, length, ferro_prob, gen, threads)

    #Snapshots of lattice
    lattice_list[0] = lattice
    energy_list[0] = calc_hamiltonian(lattice, j_matrix, threads)
    magnetisation_list[0] = calc_magnetisation(lattice, threads)

    #For timing purposes
    init_time = openmp.omp_get_wtime()

    for nround in range(nrounds):
        for offset in range(stencil):
            with nogil, parallel(num_threads=threads):
                thread = openmp.omp_get_thread_num()

                distr = uniform_int_distribution[int](rows*(stencil*thread+offset),
                                                    rows*(stencil*thread+offset+1)-1)
                
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
                        
        #Take snapshot every rounds of flips
        lattice_list[nround+1] = lattice
        energy_list[nround+1] = calc_hamiltonian(lattice, j_matrix, threads)
        magnetisation_list[nround+1] = calc_magnetisation(lattice, threads)

    final_time = openmp.omp_get_wtime()
    print("Elapsed time: %f"%(final_time-init_time))

    #Save results
    np.savez_compressed('results', length=length,
                        iterations=steps, rounds=rounds,
                        temperature=temperature, ferro_prob=ferro_prob,
                        threads=threads,
                        lattice_list=lattice_list,
                        energy_list=energy_list,
                        magnetisation_list=magnetisation_list,
                        j_matrix=j_matrix)

    #Free memory allocated to gen through PyMem_Malloc
    PyMem_Free(gen)
    
    return