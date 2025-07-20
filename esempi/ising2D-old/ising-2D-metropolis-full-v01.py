#!/usr/bin/env python
#
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def initialize_lattice(L, state='random'):
    """
    Inizializza il reticolo di spin L×L
    Args:
        L: dimensione del reticolo
        state: 'random', 'up', 'down', 'cold'
    Returns:
        lattice: array numpy L×L con spin ±1
    """
    if state == 'random':
        return np.random.choice([-1, 1], size=(L, L))
    elif state == 'up':
        return np.ones((L, L), dtype=int)
    elif state == 'down':
        return -np.ones((L, L), dtype=int)
    else:
        raise ValueError("State must be 'random', 'up', 'down'")

def local_energy(lattice, i, j, J=1.0):
    """
    Calcola l'energia locale di un singolo spin
    Args:
        lattice: reticolo di spin
        i, j: coordinate del spin
        J: costante di accoppiamento
    Returns:
        energy: energia locale del spin
    """
    L = lattice.shape[0]
    # Condizioni al contorno periodiche
    neighbors_sum = (lattice[(i-1) % L, j] + 
                    lattice[(i+1) % L, j] + 
                    lattice[i, (j-1) % L] + 
                    lattice[i, (j+1) % L])
    return -J * lattice[i, j] * neighbors_sum

def metropolis_step(lattice, beta, J=1.0):
    """
    Esegue un singolo passo dell'algoritmo di Metropolis
    Args:
        lattice: reticolo di spin (modificato in-place)
        beta: temperatura inversa (1/T)
        J: costante di accoppiamento
    Returns:
        accepted: True se il flip è stato accettato
    """
    L = lattice.shape[0]
    # Seleziona posizione casuale
    i = np.random.randint(0, L)
    j = np.random.randint(0, L)
    # Calcola energia prima del flip
    energy_before = local_energy(lattice, i, j, J)
    # Energia dopo il flip
    energy_after = -energy_before
    delta_E = energy_after - energy_before
    # Criterio di Metropolis
    if delta_E <= 0:
        # Accetta sempre se l'energia diminuisce
        lattice[i, j] *= -1
        return True
    else:
        # Accetta con probabilità exp(-β*ΔE)
        if np.random.random() < np.exp(-beta * delta_E):
            lattice[i, j] *= -1
            return True
        else:
            return False


def monte_carlo_sweep(lattice, beta, J=1.0):
    """
    Esegue una sweep completa (L² tentativi di flip)
    Args:
        lattice: reticolo di spin
        beta: temperatura inversa
        J: costante di accoppiamento
    Returns:
        acceptance_rate: frazione di flip accettati
    """
    L = lattice.shape[0]
    accepted = 0
    for _ in range(L * L):
        if metropolis_step(lattice, beta, J):
            accepted += 1
    return accepted / (L * L)



def thermodynamic_quantities(energies, magnetizations, beta, N):
    """
    Calcola le quantità termodinamiche dalle serie temporali
    Args:
        energies: array delle energie
        magnetizations: array delle magnetizzazioni
        beta: temperatura inversa
        N: numero totale di spin
    Returns:
        dict: dizionario con le quantità termodinamiche
    """
    energies = np.array(energies)
    magnetizations = np.array(magnetizations)
    mag_abs = np.abs(magnetizations)
    # Medie
    avg_energy = np.mean(energies)
    avg_mag = np.mean(mag_abs)
    # Fluttuazioni
    energy_var = np.var(energies)
    mag_var = np.var(magnetizations)
    # Quantità termodinamiche
    specific_heat = beta**2 * energy_var / N
    susceptibility = beta * N * mag_var
    return {
        'energy': avg_energy / N,
        'magnetization': avg_mag,
        'specific_heat': specific_heat,
        'susceptibility': susceptibility,
        'energy_error': np.std(energies) / np.sqrt(len(energies)) / N,
        'mag_error': np.std(mag_abs) / np.sqrt(len(mag_abs))
    }



def simulate_temperature(T, L=20, n_thermal=1000, n_measure=2000, 
                        J=1.0, initial_state='random', verbose=False):
    """
    Simula il sistema a una temperatura specifica
    Args:
        T: temperatura
        L: dimensione del reticolo
        n_thermal: passi di termalizzazione
        n_measure: passi di misurazione
        J: costante di accoppiamento
        initial_state: stato iniziale del reticolo
        verbose: stampa informazioni durante la simulazione
    Returns:
        dict: risultati della simulazione
    """
    beta = 1.0 / T
    # Inizializza il reticolo
    lattice = initialize_lattice(L, initial_state)
    # Fase di termalizzazione
    for step in range(n_thermal):
        monte_carlo_sweep(lattice, beta, J)
    # Fase di misurazione
    energies = []
    magnetizations = []
    for step in range(n_measure):
        monte_carlo_sweep(lattice, beta, J)
        energies.append(total_energy(lattice, J))
        magnetizations.append(magnetization(lattice))
    # Calcola quantità termodinamiche
    results = thermodynamic_quantities(energies, magnetizations, beta, L*L)
    return results


def find_critical_temperature(temperatures, specific_heats, susceptibilities):
    """
    Determina la temperatura critica dai picchi di C e χ
    Args:
        temperatures: array delle temperature
        specific_heats: array delle capacità termiche
        susceptibilities: array delle suscettibilità
    Returns:
        dict: temperature critiche determinate
    """
    # Temperatura critica dal picco di capacità termica
    c_max_idx = np.argmax(specific_heats)
    T_c_heat = temperatures[c_max_idx]
    # Temperatura critica dal picco di suscettibilità
    chi_max_idx = np.argmax(susceptibilities)
    T_c_suscept = temperatures[chi_max_idx]
    # Media pesata
    T_c_avg = (2 * T_c_heat + T_c_suscept) / 3
    return {
        'T_c_from_heat': T_c_heat,
        'T_c_from_suscept': T_c_suscept,
        'T_c_average': T_c_avg,
        'heat_peak_value': specific_heats[c_max_idx],
        'suscept_peak_value': susceptibilities[chi_max_idx]
    }



def plot_thermodynamic_quantities(results, save_figure=False):
    """
    Crea grafici delle quantità termodinamiche
    Args:
        results: risultati della scansione in temperatura
        save_figure: se salvare la figura
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Modello di Ising 2D - Simulazione Monte Carlo', fontsize=16)
    T = results['temperatures']
    T_c_theory = 2.269
    # Energia
    axes[0, 0].errorbar(T, results['energies'], yerr=results['energy_errors'],
                       fmt='o-', color='blue', markersize=4, capsize=3)
    axes[0, 0].axvline(T_c_theory, color='red', linestyle='--', alpha=0.7,
                      label=f'T_c teorica = {T_c_theory:.3f}')
    axes[0, 0].set_xlabel('Temperatura')
    axes[0, 0].set_ylabel('Energia per spin')
    axes[0, 0].set_title('Energia')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    # Magnetizzazione
    axes[0, 1].errorbar(T, results['magnetizations'], yerr=results['mag_errors'],
                       fmt='o-', color='green', markersize=4, capsize=3)
    axes[0, 1].axvline(T_c_theory, color='red', linestyle='--', alpha=0.7)
    axes[0, 1].set_xlabel('Temperatura')
    axes[0, 1].set_ylabel('Magnetizzazione per spin')
    axes[0, 1].set_title('Magnetizzazione')
    axes[0, 1].grid(True, alpha=0.3)
    # Capacità termica
    axes[1, 0].plot(T, results['specific_heats'], 'o-', color='purple',
                   markersize=4)
    axes[1, 0].axvline(T_c_theory, color='red', linestyle='--', alpha=0.7)
    axes[1, 0].set_xlabel('Temperatura')
    axes[1, 0].set_ylabel('Capacità termica per spin')
    axes[1, 0].set_title('Capacità Termica')
    axes[1, 0].grid(True, alpha=0.3)
    # Suscettibilità
    axes[1, 1].plot(T, results['susceptibilities'], 'o-', color='orange',
                   markersize=4)
    axes[1, 1].axvline(T_c_theory, color='red', linestyle='--', alpha=0.7)
    axes[1, 1].set_xlabel('Temperatura')
    axes[1, 1].set_ylabel('Suscettibilità per spin')
    axes[1, 1].set_title('Suscettibilità Magnetica')
    axes[1, 1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()



def plot_spin_configurations(temperatures, L=20):
    """
    Mostra configurazioni di spin a diverse temperature
    Args:
        temperatures: lista delle temperature da mostrare
        L: dimensione del reticolo
    """
    n_temps = len(temperatures)
    fig, axes = plt.subplots(1, n_temps, figsize=(4*n_temps, 4))
    for i, T in enumerate(temperatures):
        # Simula alla temperatura specifica
        results = simulate_temperature(T, L, 500, 100, verbose=False)
        lattice = results['final_lattice']
        # Mostra configurazione
        axes[i].imshow(lattice, cmap='RdBu', vmin=-1, vmax=1)
        axes[i].set_title(f'T = {T:.2f}\n'
                         f'M = {results["magnetization"]:.3f}\n'
                         f'E = {results["energy"]:.3f}')
        axes[i].set_xticks([])
        axes[i].set_yticks([])
    plt.tight_layout()
    plt.show()


T = 1.0

simulate_temperature(T, L=20, n_thermal=1000, n_measure=2000, J=1.0, initial_state='random', verbose=False)



