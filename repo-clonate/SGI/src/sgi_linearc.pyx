#!~/anaconda3/bin/python

# cython: language_level = 3
# distutils: language = c++
# distutils: extra_compile_args = -std=c++11

import time
import numpy as np


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


cdef double calc_hamiltonian(int[:,:] lattice, int[:,:] j_matrix):
    """Calculate Hamiltonian of lattice using j values in j_matrix.

    Arguments
    ---------
    lattice : 2D int array
        Lattice containing spins of +1 or -1
    j_matrix : 2D int array
        Array j values describing interactions between adjacent spins

    Returns
    -------
    H : float
        Hamiltonian
    """
    cdef:
        double H = 0.0
        int length = len(lattice)
        int i, j
    for i in range(length):
        for j in range(length):
                #left, right, above, below
                #Periodic boundary conditions
                H += -(j_matrix[2*i+1,j]*lattice[i,j]*lattice[i,j-1]
                        +j_matrix[2*i+1,(j+1)%length]*lattice[i,j]*lattice[i,(j+1)%length]
                        +j_matrix[2*i,j]*lattice[i,j]*lattice[i-1,j]
                        +j_matrix[(2*i+2)%(2*length),j]*lattice[i,j]*lattice[(i+1)%length,j])
    H = H/2
    return H


cdef double calc_h(int i, int j, int[:,:] lattice, int[:,:] j_matrix):
    """Sum H over adjacent spins of spin at position (i,j) in lattice.

    Arguments
    ---------
    i : int
        x index of spin position
    j : int
        y index of spin position
    lattice : 2D int array
        Lattice containing spins of +1 or -1
    j_matrix : 2D int array
        Array j values describing interactions between adjacent spins

    Returns
    -------
    h : double
        Sum of energy of bonds connected spin at position (ij) in lattice
    """
    cdef:
        double h
        int length = len(lattice)
    #left, right, above, below
    h = -(j_matrix[2*i+1,j]*lattice[i,j]*lattice[i,j-1]
            +j_matrix[2*i+1,(j+1)%length]*lattice[i,j]*lattice[i,(j+1)%length]
            +j_matrix[2*i,j]*lattice[i,j]*lattice[i-1,j]
            +j_matrix[(2*i+2)%(2*length),j]*lattice[i,j]*lattice[(i+1)%length,j])
    return h


cdef double calc_magnetisation(int[:,:] lattice):
    """Calculate magnetisation of lattice.

    Arguments
    ---------
    lattice : np.ndarray
        Lattice containing spins of +1 or -1
    
    Returns
    -------
    M : double
        Magnetisation of lattice
    """
    cdef:
        double M = 0.0
        int i,j
        int length = len(lattice)
    for i in range(length):
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


cpdef main(int length, int steps, int rounds, double temperature, double ferro_prob):
    
    cdef:
        int nrounds = int(steps/rounds) #No of snapshots
        int msteps = rounds #No of attempted flips before each snapshot
        int[:,::1] lattice = np.random.choice((-1, 1), size=(length, length)
                                                ).astype(np.int32)
        int[:,::1] j_matrix = np.random.choice((-1, 1), size=(length*2, length),
                                                p=[1-ferro_prob, ferro_prob]
                                                ).astype(np.int32)
        int[:,:,::1] lattice_list = np.zeros((nrounds+1, length, length),
                                                dtype=np.int32)
        int i, j, step, nround
        
        double kBT_inv
        double old_h = 0
        double new_h = 0
        double delta_h = 0
        double[:] energy_list = np.zeros(nrounds+1, dtype=float)
        double[:] magnetisation_list = np.zeros(nrounds+1, dtype=float)
        double init_time, final_time

        #RNG variables
        mt19937 gen =  mt19937(int(time.time()))    #Generator
        uniform_int_distribution[int] dist = uniform_int_distribution[int](0, length-1)
        uniform_real_distribution[double] distb = uniform_real_distribution[double](0, 1)

    #kBT_inv will be bypassed if temperature==0
    if temperature != 0.0:
        kBT_inv = 1.0/(1.3806503e-23*temperature)

    #Snapshots of lattice
    lattice_list[0] = lattice
    energy_list[0] = calc_hamiltonian(lattice, j_matrix)
    magnetisation_list[0] = calc_magnetisation(lattice)
    
    #For timing purposes
    init_time = time.time()

    for nround in range(nrounds):
        for step in range(msteps):
            #Choose random lattice site
            i = dist(gen)
            j = dist(gen)
            
            #Calculate whether spin flip at (i,j) will reduce Hamiltonian
            old_h = calc_h(i, j, lattice, j_matrix)
            lattice[i,j] = -lattice[i,j]
            new_h = calc_h(i, j, lattice, j_matrix)
            delta_h = new_h-old_h

            #Flip back spin if energy doesn't decreased
            #or Boltzmann factor isn't fulfilled
            if (not delta_h < 0):
                #Hack to bypass infinite kBT_inv if temperature==0
                if (temperature == 0.0 or
                        (not distb(gen) < exp(-delta_h*kBT_inv))):
                    lattice[i,j] = -lattice[i,j]
            
        #Take snapshot every rounds of flips
        lattice_list[nround+1] = lattice
        energy_list[nround+1] = calc_hamiltonian(lattice, j_matrix)
        magnetisation_list[nround+1] = calc_magnetisation(lattice)

    final_time = time.time()
    print("Elapsed time: %f"%(final_time-init_time))

    #Save results
    np.savez_compressed('results', length=length,
                        iterations=steps, rounds=rounds,
                        temperature=temperature, ferro_prob=ferro_prob,
                        lattice_list=lattice_list,
                        energy_list=energy_list,
                        magnetisation_list=magnetisation_list,
                        j_matrix=j_matrix)

    