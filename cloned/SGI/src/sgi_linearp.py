 #!~/anaconda3/bin/python
 
import time
import copy
import argparse
import numpy as np


def calc_hamiltonian(lattice, j_matrix):
    """Calculate Hamiltonian of lattice using j values in j_matrix.

    Arguments
    ---------
    lattice : np.ndarray
        Lattice containing spins of +1 or -1
    j_matrix : np.ndarray
        Array j values describing interactions between adjacent spins

    Returns
    -------
    H : float
        Hamiltonian
    """
    H = 0.0
    length = len(lattice)
    for i in range(length):
        for j in range(length):
            #left, right, above, below
            #Periodic boundary conditions
            H += -(j_matrix[2*i+1,j]*lattice[i,j]*lattice[i,j-1]
                    +j_matrix[2*i+1,(j+1)%length]*lattice[i,j]*lattice[i,(j+1)%length]
                    +j_matrix[2*i,j]*lattice[i,j]*lattice[i-1,j]
                    +j_matrix[(2*i+2)%(2*length),j]*lattice[i,j]*lattice[(i+1)%length,j])
    #Above sum adds each bond twice
    H = H/2
    return H


def calc_h(coefficients, lattice, j_matrix):
    """Sum H over adjacent spins of spin at coefficients in lattice.

    Arguments
    ---------
    coefficients : tuple
        2D index of spin position
    lattice : np.ndarray
        Lattice containing spins of +1 or -1
    j_matrix : np.ndarray
        Array j values describing interactions between adjacent spins

    Returns
    -------
    h : float
        Sum of energy of bonds connected spin at coefficients in lattice
    """
    i,j = coefficients
    length = len(lattice)
    #left, right, above, below
    h = -(j_matrix[2*i+1,j]*lattice[i,j]*lattice[i,j-1]
            +j_matrix[2*i+1,(j+1)%length]*lattice[i,j]*lattice[i,(j+1)%length]
            +j_matrix[2*i,j]*lattice[i,j]*lattice[i-1,j]
            +j_matrix[(2*i+2)%(2*length),j]*lattice[i,j]*lattice[(i+1)%length,j])
    return h


def calc_magnetisation(lattice):
    """Calculate magnetisation of lattice.

    Arguments
    ---------
    lattice : np.ndarray
        Lattice containing spins of +1 or -1
    
    Returns
    -------
    M : float
        Magnetisation of lattice
    """
    M = 0.0
    length = len(lattice)
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


def main(length, steps, rounds, temperature, ferro_prob):
    
    #Initialisation
    nrounds = int(steps/rounds) #No of snapshots
    msteps = rounds #No of attempted flips before each snapshot
    lattice = np.random.choice((-1, 1), size=(length, length))
    j_matrix = np.random.choice((-1, 1), size=(length*2, length),
                                p=[1-ferro_prob, ferro_prob])
    lattice_list = np.zeros((nrounds+1, length, length), dtype=np.int32)
    energy_list = np.zeros(nrounds+1, dtype=float)
    magnetisation_list = np.zeros(nrounds+1, dtype=np.double)

    #kBT_inv will be bypassed if temperature==0
    if temperature != 0.0:
        kBT_inv = 1.0/(1.3806503e-23*temperature)

    #Snapshots of lattice
    lattice_list[0] = copy.copy(lattice)
    energy_list[0] = calc_hamiltonian(lattice, j_matrix)
    magnetisation_list[0] = calc_magnetisation(lattice)

    #For timing purposes
    init_time = time.time()

    for nround in range(nrounds):
        for step in range(msteps):
            #Choose random lattice site
            i = np.random.randint(0, length)
            j = np.random.randint(0, length)
            
            #Calculate whether spin flip at (i,j) will reduce Hamiltonian
            old_h = calc_h((i,j), lattice, j_matrix)
            lattice[i,j] = -lattice[i,j]
            new_h = calc_h((i,j), lattice, j_matrix)
            delta_h = new_h-old_h
            
            #Flip back spin if energy doesn't decreased
            #or Boltzmann factor isn't fulfilled
            if (not delta_h < 0):
                #Hack to bypass infinite kBT_inv if temperature==0
                if (temperature == 0.0 or
                        (not np.random.random() < np.exp(-delta_h*kBT_inv))):
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
