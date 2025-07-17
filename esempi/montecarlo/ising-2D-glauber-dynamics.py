#!/usr/bin/env python
#
"""
DINAMICA DI GLAUBER PER IL MODELLO DI ISING 2D
=============================================

Questo codice implementa la dinamica di Glauber per il modello di Ising 2D 
utilizzando NumPy per efficienza computazionale.

Autore: Esempio per spiegazione didattica
Data: 2025
"""

#
#  Importazione librerie e funzioni
#
import numpy as np
import matplotlib.pyplot as plt
from numba import jit  # Opzionale: per ottimizzazione

# ============================================================================
# PARAMETRI GLOBALI DEL MODELLO
# ============================================================================

# Parametri fisici
J = 1.0          # Costante di accoppiamento (interazione ferromagnetica)
T = 2.0          # Temperatura del sistema
BETA = 1.0 / T   # Parametro termodinamico β = 1/(kT), k=1

# Parametri della simulazione
N = 20           # Dimensione del reticolo quadrato (N × N)
N_THERMAL = 1000 # Numero di passi di termalizzazione
N_MEASURE = 5000 # Numero di passi per la raccolta dati
MEASURE_FREQ = 5 # Frequenza di misurazione

# ============================================================================
# FUNZIONI FONDAMENTALI
# ============================================================================

def initialize_lattice(N, state='random'):
    """
    Inizializza il reticolo di spin.
    
    Parametri:
    -----------
    N : int
        Dimensione del reticolo quadrato
    state : str
        Stato iniziale: 'random', 'up', 'down'
        
    Restituisce:
    -----------
    lattice : np.ndarray
        Reticolo N×N con spin ±1
    """
    if state == 'random':
        return np.random.choice([-1, 1], size=(N, N))
    elif state == 'up':
        return np.ones((N, N), dtype=int)
    elif state == 'down':
        return -np.ones((N, N), dtype=int)
    else:
        raise ValueError("state deve essere 'random', 'up', o 'down'")

def energy_change(lattice, i, j, J=1.0):
    """
    Calcola la variazione di energia per il flip dello spin in (i,j).
    
    Parametri:
    -----------
    lattice : np.ndarray
        Reticolo di spin
    i, j : int
        Coordinate dello spin da flippare
    J : float
        Costante di accoppiamento
        
    Restituisce:
    -----------
    delta_E : float
        Variazione di energia ΔE = E_nuovo - E_vecchio
    """
    N = lattice.shape[0]
    
    # Calcola somma degli spin vicini (condizioni al contorno periodiche)
    neighbors_sum = (lattice[(i-1) % N, j] +      # sopra
                    lattice[(i+1) % N, j] +       # sotto  
                    lattice[i, (j-1) % N] +       # sinistra
                    lattice[i, (j+1) % N])        # destra
    
    # ΔE = 2 * J * spin_attuale * somma_vicini
    return 2.0 * J * lattice[i, j] * neighbors_sum

def glauber_step(lattice, beta, J=1.0):
    """
    Esegue un singolo passo della dinamica di Glauber.
    
    Algoritmo:
    1. Seleziona casualmente uno spin
    2. Calcola ΔE per il flip
    3. Accetta il flip con probabilità di Fermi p = 1/(1 + exp(βΔE))
    
    Parametri:
    -----------
    lattice : np.ndarray
        Reticolo di spin (modificato in-place)
    beta : float
        Parametro termodinamico β = 1/T
    J : float
        Costante di accoppiamento
        
    Restituisce:
    -----------
    lattice : np.ndarray
        Reticolo aggiornato
    """
    N = lattice.shape[0]
    
    # Seleziona posizione casuale
    i = np.random.randint(0, N)
    j = np.random.randint(0, N)
    
    # Calcola variazione di energia
    delta_E = energy_change(lattice, i, j, J)
    
    # Probabilità di flip (funzione di Fermi)
    flip_prob = 1.0 / (1.0 + np.exp(beta * delta_E))
    
    # Decide se flippare
    if np.random.random() < flip_prob:
        lattice[i, j] *= -1
    
    return lattice

def calculate_total_energy(lattice, J=1.0):
    """
    Calcola l'energia totale del reticolo.
    
    Hamiltonian: H = -J * Σ_{<i,j>} σ_i σ_j
    dove <i,j> indica coppie di primi vicini.
    
    Parametri:
    -----------
    lattice : np.ndarray
        Reticolo di spin
    J : float
        Costante di accoppiamento
        
    Restituisce:
    -----------
    energy : float
        Energia totale del sistema
    """
    N = lattice.shape[0]
    energy = 0.0
    
    for i in range(N):
        for j in range(N):
            # Conta solo interazioni verso destra e verso il basso
            # per evitare doppi conteggi
            right = lattice[i, (j+1) % N]
            down = lattice[(i+1) % N, j]
            energy += -J * lattice[i, j] * (right + down)
    
    return energy

def calculate_magnetization(lattice):
    """
    Calcola la magnetizzazione del reticolo.
    
    Parametri:
    -----------
    lattice : np.ndarray
        Reticolo di spin
        
    Restituisce:
    -----------
    magnetization : float
        Magnetizzazione per spin M = (1/N²) Σ σ_i
    """
    N = lattice.shape[0]
    return np.sum(lattice) / (N * N)

# ============================================================================
# SIMULAZIONE PRINCIPALE
# ============================================================================

def run_glauber_simulation(N, T, J=1.0, n_thermal=1000, n_measure=5000, 
                          measure_freq=5, initial_state='random', verbose=True):
    """
    Esegue una simulazione completa con dinamica di Glauber.
    
    Parametri:
    -----------
    N : int
        Dimensione del reticolo
    T : float
        Temperatura
    J : float
        Costante di accoppiamento
    n_thermal : int
        Passi di termalizzazione
    n_measure : int
        Passi per le misure
    measure_freq : int
        Frequenza di misurazione
    initial_state : str
        Stato iniziale del reticolo
    verbose : bool
        Stampa informazioni durante la simulazione
        
    Restituisce:
    -----------
    results : dict
        Dizionario con i risultati della simulazione
    """
    
    beta = 1.0 / T
    
    if verbose:
        print(f"=== SIMULAZIONE DINAMICA DI GLAUBER ===")
        print(f"Reticolo: {N}×{N} = {N*N} spin")
        print(f"Temperatura: T = {T}")
        print(f"Parametro β: {beta:.4f}")
        print(f"Costante J: {J}")
        print(f"Termalizzazione: {n_thermal} passi")
        print(f"Misurazioni: {n_measure} passi")
        print()
    
    # Inizializza reticolo
    lattice = initialize_lattice(N, initial_state)
    
    # FASE 1: Termalizzazione
    if verbose:
        print("TERMALIZZAZIONE...")
    
    for step in range(n_thermal):
        lattice = glauber_step(lattice, beta, J)
        
        if verbose and (step + 1) % (n_thermal // 5) == 0:
            E = calculate_total_energy(lattice, J)
            M = calculate_magnetization(lattice)
            print(f"  {step+1}/{n_thermal}: E = {E:.1f}, M = {M:.3f}")
    
    # FASE 2: Raccolta dati
    if verbose:
        print("\nRACCOLTA DATI...")
    
    energies = []
    magnetizations = []
    
    for step in range(n_measure):
        lattice = glauber_step(lattice, beta, J)
        
        if step % measure_freq == 0:
            E = calculate_total_energy(lattice, J)
            M = calculate_magnetization(lattice)
            energies.append(E)
            magnetizations.append(M)
    
    # Converti in array
    energies = np.array(energies)
    magnetizations = np.array(magnetizations)
    
    # Analisi statistica
    results = {
        'lattice': lattice,
        'energies': energies,
        'magnetizations': magnetizations,
        'energy_mean': np.mean(energies),
        'energy_std': np.std(energies),
        'energy_per_spin': np.mean(energies) / (N * N),
        'magnetization_mean': np.mean(magnetizations),
        'magnetization_std': np.std(magnetizations),
        'magnetization_abs': np.mean(np.abs(magnetizations)),
        'specific_heat': np.var(energies) * beta**2 / (N * N),
        'susceptibility': np.var(magnetizations) * beta * (N * N),
        'temperature': T,
        'beta': beta,
        'coupling': J,
        'lattice_size': N
    }
    
    if verbose:
        print(f"Raccolti {len(energies)} campioni")
        print("\n=== RISULTATI ===")
        print(f"⟨E⟩ = {results['energy_mean']:.2f} ± {results['energy_std']:.2f}")
        print(f"⟨E⟩/N = {results['energy_per_spin']:.4f}")
        print(f"⟨M⟩ = {results['magnetization_mean']:.4f} ± {results['magnetization_std']:.4f}")
        print(f"⟨|M|⟩ = {results['magnetization_abs']:.4f}")
        print(f"C = {results['specific_heat']:.4f}")
        print(f"χ = {results['susceptibility']:.4f}")
    
    return results

# ============================================================================
# ESEMPIO DI UTILIZZO
# ============================================================================

if __name__ == "__main__":
    # Parametri della simulazione
    N = 16
    T = 2.5
    J = 1.0
    
    # Esegui simulazione
    results = run_glauber_simulation(N, T, J, n_thermal=1000, n_measure=2000)
    
    # Mostra configurazione finale
    print("\n=== CONFIGURAZIONE FINALE ===")
    print(results['lattice'][:8, :8])  # Mostra solo 8x8 per chiarezza
    
    # Statistiche degli spin
    lattice = results['lattice']
    n_up = np.sum(lattice == 1)
    n_down = np.sum(lattice == -1)
    print(f"\nSpin up: {n_up}")
    print(f"Spin down: {n_down}")
    print(f"Rapporto up/down: {n_up/n_down:.3f}")

