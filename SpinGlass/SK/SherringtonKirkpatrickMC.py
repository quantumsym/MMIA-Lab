#!/usr/bin/python
#
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from scipy import stats

class SherringtonKirkpatrickMC:
    """
    Simulazione Monte Carlo del modello Sherrington-Kirkpatrick (SK) per vetri di spin
    
    Il modello SK è caratterizzato da:
    - Interazioni casuali gaussiane a lungo raggio
    - Comportamento mean-field
    - Transizione di fase verso una fase vetrosa a bassa temperatura
    """
    
    def __init__(self, N, temperature=1.0, coupling_strength=1.0):
        """
        Inizializza il sistema SK
        
        Args:
            N (int): Numero di spin nel sistema
            temperature (float): Temperatura del sistema (kT/J)
            coupling_strength (float): Intensità delle interazioni
        """
        self.N = N
        self.T = temperature
        self.beta = 1.0 / temperature if temperature > 0 else float('inf')
        self.J = coupling_strength
        
        # Inizializza spin casuali
        self.spins = np.random.choice([-1, 1], size=N)
        
        # Genera matrice delle interazioni casuali (simmetrica)
        self.interactions = self._generate_interaction_matrix()
        
        # Statistiche della simulazione
        self.energy_history = []
        self.magnetization_history = []
        self.overlap_history = []
        self.susceptibility_history = []
        
    def _generate_interaction_matrix(self):
        """
        Genera la matrice delle interazioni gaussiane per il modello SK
        """
        # Matrice triangolare superiore con elementi gaussiani
        interactions = np.zeros((self.N, self.N))
        
        # Riempi triangolo superiore con numeri gaussiani
        for i in range(self.N):
            for j in range(i + 1, self.N):
                interactions[i, j] = np.random.normal(0, self.J / np.sqrt(self.N))
                interactions[j, i] = interactions[i, j]  # Simmetria
        
        return interactions
    
    def energy(self, spins=None):
        """
        Calcola l'energia del sistema secondo l'Hamiltoniana SK
        """
        if spins is None:
            spins = self.spins
        
        return -0.5 * np.sum(self.interactions * np.outer(spins, spins))
    
    def magnetization(self, spins=None):
        """
        Calcola la magnetizzazione del sistema
        """
        if spins is None:
            spins = self.spins
        return np.mean(spins)
    
    def overlap(self, spins1, spins2):
        """
        Calcola il parametro di overlap tra due configurazioni
        """
        return np.mean(spins1 * spins2)
    
    def local_field(self, site):
        """
        Calcola il campo locale su un sito specifico
        """
        return np.sum(self.interactions[site] * self.spins) - \
               self.interactions[site, site] * self.spins[site]
    
    def metropolis_step(self):
        """
        Esegue un singolo passo dell'algoritmo di Metropolis
        """
        # Seleziona spin casuale
        site = np.random.randint(self.N)
        
        # Calcola variazione di energia per flip dello spin
        h_local = self.local_field(site)
        delta_E = 2 * self.spins[site] * h_local
        
        # Criterio di accettazione di Metropolis
        if delta_E <= 0 or np.random.random() < np.exp(-self.beta * delta_E):
            self.spins[site] *= -1
            return True
        
        return False
    
    def heat_bath_step(self):
        """
        Esegue un passo con l'algoritmo heat bath (più efficiente a bassa T)
        """
        site = np.random.randint(self.N)
        h_local = self.local_field(site)
        
        # Probabilità di avere spin up
        prob_up = 1.0 / (1.0 + np.exp(-2 * self.beta * h_local))
        
        # Aggiorna spin secondo probabilità heat bath
        self.spins[site] = 1 if np.random.random() < prob_up else -1
    
    def simulate(self, n_steps, algorithm='metropolis', equilibration_steps=None):
        """
        Esegue la simulazione Monte Carlo
        
        Args:
            n_steps (int): Numero di passi Monte Carlo
            algorithm (str): 'metropolis' o 'heat_bath'
            equilibration_steps (int): Passi di equilibrazione (default: n_steps//10)
        """
        if equilibration_steps is None:
            equilibration_steps = n_steps // 10
        
        # Reset delle statistiche
        self.energy_history = []
        self.magnetization_history = []
        
        # Scegli algoritmo
        step_function = self.metropolis_step if algorithm == 'metropolis' else self.heat_bath_step
        
        print(f"🔥 Equilibrazione per {equilibration_steps} passi...")
        
        # Fase di equilibrazione
        for _ in tqdm(range(equilibration_steps), desc="Equilibrazione"):
            for _ in range(self.N):  # Un sweep = N tentativi
                step_function()
        
        print(f"📊 Simulazione produttiva per {n_steps} passi...")
        
        # Fase di produzione con raccolta dati
        accepted_moves = 0
        total_moves = 0
        
        for step in tqdm(range(n_steps), desc="Simulazione"):
            # Un sweep Monte Carlo
            step_accepted = 0
            for _ in range(self.N):
                if algorithm == 'metropolis':
                    if self.metropolis_step():
                        step_accepted += 1
                else:
                    self.heat_bath_step()
                    step_accepted += 1
                
                total_moves += 1
            
            accepted_moves += step_accepted
            
            # Raccogli statistiche ogni 10 sweep
            if step % 10 == 0:
                self.energy_history.append(self.energy())
                self.magnetization_history.append(self.magnetization())
        
        acceptance_rate = accepted_moves / total_moves if algorithm == 'metropolis' else 1.0
        
        return {
            'acceptance_rate': acceptance_rate,
            'final_energy': self.energy(),
            'final_magnetization': self.magnetization(),
            'energy_history': np.array(self.energy_history),
            'magnetization_history': np.array(self.magnetization_history)
        }
    
    def simulated_annealing(self, T_initial, T_final, cooling_rate=0.99, 
                           steps_per_temp=1000):
        """
        Implementa simulated annealing per trovare configurazioni a bassa energia
        """
        self.T = T_initial
        self.beta = 1.0 / self.T
        
        temperature_history = []
        energy_history = []
        
        print(f"❄️ Simulated Annealing: T = {T_initial:.3f} → {T_final:.3f}")
        
        while self.T > T_final:
            # Simula a temperatura costante
            for _ in range(steps_per_temp):
                self.metropolis_step()
            
            # Raccogli statistiche
            temperature_history.append(self.T)
            energy_history.append(self.energy())
            
            # Raffredda il sistema
            self.T *= cooling_rate
            self.beta = 1.0 / self.T
        
        return {
            'final_temperature': self.T,
            'final_energy': self.energy(),
            'temperature_history': temperature_history,
            'energy_history': energy_history
        }
    
    def parallel_tempering(self, temperatures, n_steps, exchange_interval=100):
        """
        Implementa parallel tempering (replica exchange) per migliorare l'esplorazione
        """
        n_replicas = len(temperatures)
        
        # Inizializza repliche
        replicas = []
        for T in temperatures:
            replica = SherringtonKirkpatrickMC(self.N, T, self.J)
            replica.interactions = self.interactions.copy()  # Stessa realizzazione del disordine
            replicas.append(replica)
        
        # Storia per ogni replica
        energy_histories = [[] for _ in range(n_replicas)]
        exchange_attempts = 0
        exchange_accepts = 0
        
        print(f"🔄 Parallel Tempering con {n_replicas} repliche...")
        
        for step in tqdm(range(n_steps), desc="PT Simulation"):
            # Passo Monte Carlo per ogni replica
            for i, replica in enumerate(replicas):
                replica.metropolis_step()
                
                # Raccogli statistiche
                if step % 10 == 0:
                    energy_histories[i].append(replica.energy())
            
            # Tentativo di scambio tra repliche adiacenti
            if step % exchange_interval == 0 and step > 0:
                for i in range(n_replicas - 1):
                    exchange_attempts += 1
                    
                    # Calcola probabilità di scambio
                    beta_i = 1.0 / temperatures[i]
                    beta_j = 1.0 / temperatures[i + 1]
                    E_i = replicas[i].energy()
                    E_j = replicas[i + 1].energy()
                    
                    delta_beta = beta_i - beta_j
                    delta_E = E_i - E_j
                    
                    if np.random.random() < np.exp(delta_beta * delta_E):
                        # Scambia configurazioni
                        replicas[i].spins, replicas[i + 1].spins = \
                            replicas[i + 1].spins.copy(), replicas[i].spins.copy()
                        exchange_accepts += 1
        
        exchange_rate = exchange_accepts / exchange_attempts if exchange_attempts > 0 else 0
        
        return {
            'replicas': replicas,
            'energy_histories': energy_histories,
            'exchange_rate': exchange_rate,
            'temperatures': temperatures
        }
    
    def compute_observables(self, n_measurements=1000):
        """
        Calcola osservabili termodinamiche del sistema
        """
        energies = []
        magnetizations = []
        
        # Raccolta dati
        for _ in range(n_measurements):
            # Alcuni passi tra le misure per decorrelazione
            for _ in range(10):
                self.metropolis_step()
            
            energies.append(self.energy())
            magnetizations.append(self.magnetization())
        
        energies = np.array(energies)
        magnetizations = np.array(magnetizations)
        
        # Calcola osservabili
        avg_energy = np.mean(energies)
        avg_magnetization = np.mean(magnetizations)
        
        # Calore specifico
        specific_heat = self.beta**2 * (np.var(energies))
        
        # Suscettibilità magnetica
        susceptibility = self.beta * self.N * np.var(magnetizations)
        
        # Parametro di overlap con configurazione iniziale
        initial_config = np.random.choice([-1, 1], size=self.N)
        overlap_values = []
        
        for _ in range(100):
            for _ in range(10):
                self.metropolis_step()
            overlap_values.append(self.overlap(self.spins, initial_config))
        
        return {
            'energy': avg_energy,
            'magnetization': avg_magnetization,
            'specific_heat': specific_heat,
            'susceptibility': susceptibility,
            'overlap_distribution': np.array(overlap_values)
        }
    
    def plot_results(self, results):
        """
        Visualizza i risultati della simulazione
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Simulazione SK Model (N={self.N}, T={self.T:.3f})', fontsize=16)
        
        # Energia vs tempo
        if 'energy_history' in results:
            axes[0, 0].plot(results['energy_history'], 'b-', linewidth=1)
            axes[0, 0].set_xlabel('Tempo Monte Carlo')
            axes[0, 0].set_ylabel('Energia')
            axes[0, 0].set_title('Evoluzione Energia')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Magnetizzazione vs tempo
        if 'magnetization_history' in results:
            axes[0, 1].plot(results['magnetization_history'], 'r-', linewidth=1)
            axes[0, 1].set_xlabel('Tempo Monte Carlo')
            axes[0, 1].set_ylabel('Magnetizzazione')
            axes[0, 1].set_title('Evoluzione Magnetizzazione')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Istogramma energia
        if 'energy_history' in results:
            final_energies = results['energy_history'][-100:]  # Ultimi 100 punti
            axes[1, 0].hist(final_energies, bins=20, alpha=0.7, color='blue', density=True)
            axes[1, 0].set_xlabel('Energia')
            axes[1, 0].set_ylabel('Densità di Probabilità')
            axes[1, 0].set_title('Distribuzione Energia Finale')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Istogramma magnetizzazione
        if 'magnetization_history' in results:
            final_mags = results['magnetization_history'][-100:]
            axes[1, 1].hist(final_mags, bins=20, alpha=0.7, color='red', density=True)
            axes[1, 1].set_xlabel('Magnetizzazione')
            axes[1, 1].set_ylabel('Densità di Probabilità')
            axes[1, 1].set_title('Distribuzione Magnetizzazione')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


