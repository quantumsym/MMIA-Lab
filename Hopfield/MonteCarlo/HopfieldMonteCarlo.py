#!/usr/bin/env python
#
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

class HopfieldMonteCarlo:
    def __init__(self, n_neurons, temperature=1.0, cooling_rate=0.95):
        """
        Rete di Hopfield con dinamica Monte Carlo
        
        Args:
            n_neurons (int): Numero di neuroni nella rete
            temperature (float): Temperatura iniziale del sistema
            cooling_rate (float): Tasso di raffreddamento per simulated annealing
        """
        self.n_neurons = n_neurons
        self.temperature = temperature
        self.initial_temperature = temperature
        self.cooling_rate = cooling_rate
        self.weights = np.zeros((n_neurons, n_neurons))
        self.patterns = []
        
    def train(self, patterns):
        """
        Addestra la rete utilizzando la regola di Hebb
        
        Args:
            patterns (list): Lista di pattern binari (-1, 1)
        """
        self.patterns = patterns
        self.weights = np.zeros((self.n_neurons, self.n_neurons))
        
        for pattern in patterns:
            p = np.where(pattern >= 0, 1, -1).flatten()
            self.weights += np.outer(p, p)
        
        # Imposta diagonale a zero e normalizza
        np.fill_diagonal(self.weights, 0)
        self.weights /= len(patterns)
    
    def energy(self, state):
        """
        Calcola l'energia del sistema secondo il modello di Hopfield
        E = -0.5 * Σᵢⱼ wᵢⱼ sᵢ sⱼ
        
        Args:
            state (np.array): Stato corrente della rete
            
        Returns:
            float: Energia del sistema
        """
        return -0.5 * np.dot(state, np.dot(self.weights, state))
    
    def boltzmann_probability(self, delta_energy):
        """
        Calcola la probabilità di accettazione secondo la distribuzione di Boltzmann
        P(accettazione) = exp(-ΔE / kT)
        
        Args:
            delta_energy (float): Variazione di energia
            
        Returns:
            float: Probabilità di accettazione
        """
        if delta_energy <= 0:
            return 1.0
        return np.exp(-delta_energy / self.temperature)
    
    def metropolis_step(self, state):
        """
        Esegue un singolo passo dell'algoritmo di Metropolis
        
        Args:
            state (np.array): Stato corrente
            
        Returns:
            tuple: (nuovo_stato, energia_attuale, accettato)
        """
        # Seleziona neurone casuale
        neuron_idx = np.random.randint(self.n_neurons)
        
        # Calcola energia attuale
        current_energy = self.energy(state)
        
        # Propone cambio di stato (flip del neurone)
        new_state = state.copy()
        new_state[neuron_idx] *= -1
        
        # Calcola nuova energia
        new_energy = self.energy(new_state)
        delta_energy = new_energy - current_energy
        
        # Criterio di accettazione di Metropolis
        if np.random.random() < self.boltzmann_probability(delta_energy):
            return new_state, new_energy, True
        else:
            return state, current_energy, False
    
    def simulate_annealing(self, initial_state, max_iterations=10000, 
                          min_temperature=0.01, verbose=True):
        """
        Simulazione Monte Carlo con simulated annealing
        
        Args:
            initial_state (np.array): Stato iniziale
            max_iterations (int): Numero massimo di iterazioni
            min_temperature (float): Temperatura minima
            verbose (bool): Mostra progress bar
            
        Returns:
            dict: Risultati della simulazione
        """
        state = np.where(initial_state.flatten() >= 0, 1, -1)
        current_energy = self.energy(state)
        
        # Storia della simulazione
        energy_history = [current_energy]
        temperature_history = [self.temperature]
        acceptance_history = []
        state_history = [state.copy()]
        
        best_state = state.copy()
        best_energy = current_energy
        
        iterator = tqdm(range(max_iterations)) if verbose else range(max_iterations)
        accepted_moves = 0
        
        for iteration in iterator:
            # Passo Metropolis
            new_state, new_energy, accepted = self.metropolis_step(state)
            
            if accepted:
                state = new_state
                current_energy = new_energy
                accepted_moves += 1
                
                # Aggiorna migliore soluzione
                if current_energy < best_energy:
                    best_state = state.copy()
                    best_energy = current_energy
            
            # Raffreddamento
            if self.temperature > min_temperature:
                self.temperature *= self.cooling_rate
            
            # Salva statistiche ogni 100 iterazioni
            if iteration % 100 == 0:
                energy_history.append(current_energy)
                temperature_history.append(self.temperature)
                acceptance_history.append(accepted_moves / (iteration + 1))
                state_history.append(state.copy())
            
            if verbose and iteration % 1000 == 0:
                iterator.set_description(
                    f"E: {current_energy:.3f}, T: {self.temperature:.3f}, "
                    f"Acc: {accepted_moves/(iteration+1):.3f}"
                )
        
        return {
            'final_state': state,
            'best_state': best_state,
            'best_energy': best_energy,
            'energy_history': energy_history,
            'temperature_history': temperature_history,
            'acceptance_history': acceptance_history,
            'state_history': state_history,
            'total_accepted': accepted_moves,
            'acceptance_rate': accepted_moves / max_iterations
        }
    
    def parallel_tempering(self, initial_state, temperatures, max_iterations=5000):
        """
        Implementa Parallel Tempering (Replica Exchange Monte Carlo)
        
        Args:
            initial_state (np.array): Stato iniziale
            temperatures (list): Lista delle temperature per le repliche
            max_iterations (int): Numero di iterazioni
            
        Returns:
            dict: Risultati del parallel tempering
        """
        n_replicas = len(temperatures)
        states = [np.where(initial_state.flatten() >= 0, 1, -1) for _ in range(n_replicas)]
        energies = [self.energy(state) for state in states]
        
        # Storia per ogni replica
        energy_histories = [[] for _ in range(n_replicas)]
        exchange_attempts = 0
        exchange_accepts = 0
        
        for iteration in tqdm(range(max_iterations), desc="Parallel Tempering"):
            # Passo Monte Carlo per ogni replica
            for i in range(n_replicas):
                self.temperature = temperatures[i]
                new_state, new_energy, _ = self.metropolis_step(states[i])
                states[i] = new_state
                energies[i] = new_energy
                energy_histories[i].append(energies[i])
            
            # Tentativo di scambio tra repliche adiacenti
            if iteration % 10 == 0:  # Ogni 10 iterazioni
                for i in range(n_replicas - 1):
                    exchange_attempts += 1
                    
                    # Probabilità di scambio
                    beta_i = 1.0 / temperatures[i]
                    beta_j = 1.0 / temperatures[i + 1]
                    delta = (beta_i - beta_j) * (energies[i] - energies[i + 1])
                    
                    if np.random.random() < np.exp(min(0, delta)):
                        # Scambia stati
                        states[i], states[i + 1] = states[i + 1].copy(), states[i].copy()
                        energies[i], energies[i + 1] = energies[i + 1], energies[i]
                        exchange_accepts += 1
        
        return {
            'states': states,
            'energies': energies,
            'energy_histories': energy_histories,
            'temperatures': temperatures,
            'exchange_rate': exchange_accepts / exchange_attempts if exchange_attempts > 0 else 0
        }
    
    def analyze_basin_attraction(self, target_pattern, noise_levels, 
                               n_trials=50, max_iterations=1000):
        """
        Analizza i bacini di attrazione con diversi livelli di rumore
        
        Args:
            target_pattern (np.array): Pattern target
            noise_levels (list): Livelli di rumore da testare
            n_trials (int): Numero di prove per livello
            max_iterations (int): Iterazioni massime per prova
            
        Returns:
            dict: Statistiche sui bacini di attrazione
        """
        results = {'noise_levels': noise_levels, 'success_rates': [], 'avg_energies': []}
        
        for noise_level in tqdm(noise_levels, desc="Analyzing basins"):
            successes = 0
            final_energies = []
            
            for trial in range(n_trials):
                # Crea pattern rumoroso
                noisy_pattern = target_pattern.copy()
                n_flips = int(len(target_pattern) * noise_level)
                flip_indices = np.random.choice(len(target_pattern), n_flips, replace=False)
                noisy_pattern[flip_indices] *= -1
                
                # Simulazione
                self.temperature = self.initial_temperature
                result = self.simulate_annealing(
                    noisy_pattern, max_iterations=max_iterations, verbose=False
                )
                
                # Verifica successo (overlap > 0.9 con pattern originale)
                overlap = np.dot(result['best_state'], target_pattern) / len(target_pattern)
                if overlap > 0.9:
                    successes += 1
                
                final_energies.append(result['best_energy'])
            
            results['success_rates'].append(successes / n_trials)
            results['avg_energies'].append(np.mean(final_energies))
        
        return results
    
    def plot_results(self, result, title="Simulazione Monte Carlo"):
        """
        Visualizza i risultati della simulazione
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(title, fontsize=16)
        
        # Energia vs iterazioni
        axes[0, 0].plot(result['energy_history'], 'b-', linewidth=2)
        axes[0, 0].set_xlabel('Iterazioni (×100)')
        axes[0, 0].set_ylabel('Energia')
        axes[0, 0].set_title('Evoluzione Energia')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Temperatura vs iterazioni
        axes[0, 1].plot(result['temperature_history'], 'r-', linewidth=2)
        axes[0, 1].set_xlabel('Iterazioni (×100)')
        axes[0, 1].set_ylabel('Temperatura')
        axes[0, 1].set_title('Raffreddamento')
        axes[0, 1].set_yscale('log')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Tasso di accettazione
        if len(result['acceptance_history']) > 1:
            axes[1, 0].plot(result['acceptance_history'], 'g-', linewidth=2)
            axes[1, 0].set_xlabel('Iterazioni (×100)')
            axes[1, 0].set_ylabel('Tasso Accettazione')
            axes[1, 0].set_title('Dinamica Accettazione')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Distribuzione energia finale
        final_energies = result['energy_history'][-20:]  # Ultime 20 misure
        axes[1, 1].hist(final_energies, bins=10, alpha=0.7, color='purple')
        axes[1, 1].axvline(result['best_energy'], color='red', linestyle='--', 
                          label=f'Migliore: {result["best_energy"]:.3f}')
        axes[1, 1].set_xlabel('Energia')
        axes[1, 1].set_ylabel('Frequenza')
        axes[1, 1].set_title('Distribuzione Energia Finale')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


