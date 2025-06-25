<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" class="logo" width="120"/>

# Programma Python per la simulazione del modello Sherrington-Kirkpatrick con metodi Monte Carlo

Il modello Sherrington-Kirkpatrick (SK) è un modello mean-field per i vetri di spin, introdotto nel 1975 da David Sherrington e Scott Kirkpatrick[^1]. Questo modello rappresenta un sistema di spin di Ising con interazioni casuali a lungo raggio, caratterizzato da un comportamento magnetico disordinato e dalla presenza di frustrazioni geometriche[^2]. Ecco un'implementazione completa in Python che utilizza metodi Monte Carlo avanzati per simulare il modello SK.

## Formulazione del modello SK

L'Hamiltoniana del modello Sherrington-Kirkpatrick è definita come[^3][^4]:

\$ H_N(\sigma) = -\frac{1}{\sqrt{N}} \sum_{i,j=1}^{N} g_{ij}\sigma_i\sigma_j \$

dove $\sigma_i = \pm 1$ sono gli spin di Ising, $g_{ij}$ sono variabili casuali gaussiane indipendenti con media zero e varianza unitaria, e la normalizzazione $1/\sqrt{N}$ mantiene l'energia per spin di ordine 1[^5][^4].

## Implementazione completa

```python
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


# Esempio di utilizzo e test del modello
def main_simulation():
    """
    Esempio completo di simulazione del modello SK
    """
    print("🧠 Simulazione del modello Sherrington-Kirkpatrick")
    print("=" * 60)
    
    # Parametri del sistema
    N = 200  # Numero di spin
    T_high = 2.0  # Temperatura alta (fase paramagnetica)
    T_low = 0.5   # Temperatura bassa (fase vetrosa)
    
    # Test 1: Simulazione a temperatura alta
    print(f"\n🔥 Test 1: Simulazione a T = {T_high} (fase paramagnetica)")
    sk_high = SherringtonKirkpatrickMC(N, temperature=T_high)
    results_high = sk_high.simulate(n_steps=5000)
    
    print(f"✅ Risultati alta temperatura:")
    print(f"   • Energia finale: {results_high['final_energy']:.4f}")
    print(f"   • Magnetizzazione: {results_high['final_magnetization']:.4f}")
    print(f"   • Tasso accettazione: {results_high['acceptance_rate']:.3f}")
    
    # Test 2: Simulazione a temperatura bassa
    print(f"\n❄️ Test 2: Simulazione a T = {T_low} (fase vetrosa)")
    sk_low = SherringtonKirkpatrickMC(N, temperature=T_low)
    results_low = sk_low.simulate(n_steps=5000)
    
    print(f"✅ Risultati bassa temperatura:")
    print(f"   • Energia finale: {results_low['final_energy']:.4f}")
    print(f"   • Magnetizzazione: {results_low['final_magnetization']:.4f}")
    print(f"   • Tasso accettazione: {results_low['acceptance_rate']:.3f}")
    
    # Test 3: Simulated Annealing
    print(f"\n🌡️ Test 3: Simulated Annealing")
    sk_annealing = SherringtonKirkpatrickMC(N, temperature=3.0)
    sa_results = sk_annealing.simulated_annealing(
        T_initial=3.0, 
        T_final=0.1, 
        cooling_rate=0.95, 
        steps_per_temp=500
    )
    
    print(f"✅ Risultati Simulated Annealing:")
    print(f"   • Energia finale: {sa_results['final_energy']:.4f}")
    print(f"   • Temperatura finale: {sa_results['final_temperature']:.4f}")
    
    # Test 4: Parallel Tempering
    print(f"\n🔄 Test 4: Parallel Tempering")
    temperatures = [0.5, 0.8, 1.2, 1.8, 2.5]
    sk_pt = SherringtonKirkpatrickMC(N, temperature=1.0)
    pt_results = sk_pt.parallel_tempering(
        temperatures=temperatures,
        n_steps=3000,
        exchange_interval=50
    )
    
    print(f"✅ Risultati Parallel Tempering:")
    print(f"   • Tasso di scambio: {pt_results['exchange_rate']:.3f}")
    
    # Trova replica a temperatura più bassa
    lowest_temp_idx = 0
    lowest_energy = pt_results['replicas'][lowest_temp_idx].energy()
    print(f"   • Energia minima (T={temperatures[lowest_temp_idx]}): {lowest_energy:.4f}")
    
    # Test 5: Calcolo osservabili termodinamiche
    print(f"\n📊 Test 5: Osservabili termodinamiche (T = {T_low})")
    observables = sk_low.compute_observables(n_measurements=1000)
    
    print(f"✅ Osservabili:")
    print(f"   • Energia media: {observables['energy']:.4f}")
    print(f"   • Magnetizzazione media: {observables['magnetization']:.4f}")
    print(f"   • Calore specifico: {observables['specific_heat']:.4f}")
    print(f"   • Suscettibilità: {observables['susceptibility']:.4f}")
    print(f"   • Overlap medio: {np.mean(observables['overlap_distribution']):.4f}")
    
    # Visualizzazione risultati
    print(f"\n📈 Generazione grafici...")
    sk_low.plot_results(results_low)
    
    # Plot confronto temperature
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(results_high['energy_history'], 'r-', label=f'T = {T_high}', alpha=0.7)
    plt.plot(results_low['energy_history'], 'b-', label=f'T = {T_low}', alpha=0.7)
    plt.xlabel('Tempo MC')
    plt.ylabel('Energia')
    plt.title('Confronto Energie')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.plot(results_high['magnetization_history'], 'r-', label=f'T = {T_high}', alpha=0.7)
    plt.plot(results_low['magnetization_history'], 'b-', label=f'T = {T_low}', alpha=0.7)
    plt.xlabel('Tempo MC')
    plt.ylabel('Magnetizzazione')
    plt.title('Confronto Magnetizzazioni')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    plt.plot(sa_results['temperature_history'], sa_results['energy_history'], 'g-o', markersize=3)
    plt.xlabel('Temperatura')
    plt.ylabel('Energia')
    plt.title('Simulated Annealing')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    plt.hist(observables['overlap_distribution'], bins=20, alpha=0.7, color='purple', density=True)
    plt.xlabel('Overlap')
    plt.ylabel('Densità')
    plt.title('Distribuzione Overlap')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\n🎉 Simulazione completata con successo!")
    print("=" * 60)

if __name__ == "__main__":
    main_simulation()
```


## Caratteristiche del programma

### **Algoritmi Monte Carlo implementati**

Il programma include tre algoritmi Monte Carlo fondamentali per la simulazione di vetri di spin[^6][^7]:

1. **Algoritmo di Metropolis**: Implementa il criterio di accettazione standard per esplorare lo spazio delle configurazioni con probabilità di accettazione P = min(1, exp(-βΔE))[^8]
2. **Heat Bath**: Algoritmo più efficiente a basse temperature che campiona direttamente dalla distribuzione di Boltzmann locale[^8]
3. **Simulated Annealing**: Tecnica di ottimizzazione che raffredda gradualmente il sistema per trovare configurazioni a bassa energia[^9][^10]
4. **Parallel Tempering**: Metodo avanzato che simula multiple copie del sistema a temperature diverse, permettendo scambi periodici per migliorare l'esplorazione[^11]

### **Osservabili fisiche calcolate**

Il programma calcola automaticamente le principali grandezze termodinamiche caratteristiche dei vetri di spin[^2][^12]:

- **Energia media**: Valore dell'Hamiltoniana nel regime di equilibrio
- **Magnetizzazione**: Momento magnetico medio del sistema
- **Calore specifico**: Fluttuazioni dell'energia, indicatore delle transizioni di fase
- **Suscettibilità magnetica**: Risposta del sistema a campi esterni
- **Parametro di overlap**: Misura delle correlazioni tra configurazioni diverse, cruciale per caratterizzare la fase vetrosa[^13][^14]


### **Analisi delle transizioni di fase**

Il modello SK presenta una transizione di fase da una fase paramagnetica ad alta temperatura a una fase vetrosa a bassa temperatura[^2][^15]. La temperatura critica teorica è Tc = 1 (in unità adimensionali), e il programma permette di studiare questa transizione attraverso il comportamento degli osservabili.

## Vantaggi dell'implementazione

L'approccio Monte Carlo per il modello SK offre diversi vantaggi rispetto ai metodi analitici[^16][^17]:

- **Flessibilità**: Possibilità di studiare diverse realizzazioni del disordine e diversi parametri
- **Controllo completo**: Accesso a tutte le configurazioni microscopiche durante l'evoluzione
- **Metodi avanzati**: Implementazione di tecniche sofisticate come parallel tempering per superare le barriere energetiche
- **Visualizzazione**: Analisi in tempo reale dell'evoluzione del sistema e delle distribuzioni di probabilità

Questo programma fornisce una base solida per lo studio computazionale dei vetri di spin SK, permettendo di esplorare sia il regime di alta temperatura (fase paramagnetica) che quello di bassa temperatura (fase vetrosa), caratteristico di questi sistemi frustrati[^2][^4].

<div style="text-align: center">⁂</div>

[^1]: https://link.aps.org/doi/10.1103/PhysRevLett.35.1792

[^2]: https://en.wikipedia.org/wiki/Spin_glass

[^3]: https://annals.math.princeton.edu/wp-content/uploads/annals-v163-n1-p04.pdf

[^4]: http://arxiv.org/pdf/1211.1094.pdf

[^5]: https://www.intlpress.com/site/pub/files/_fulltext/journals/cdm/2014/2014/0001/CDM-2014-2014-0001-a004.pdf

[^6]: https://arxiv.org/abs/2210.11288

[^7]: https://arxiv.org/abs/2006.08378

[^8]: https://magnetism.eu/esm/2009/slides/neda-tutorial-1.pdf

[^9]: https://arxiv.org/html/2309.11822v3

[^10]: https://www.cs.amherst.edu/~ccmcgeoch/cs34/papers/parkkimsdarticle-5.pdf

[^11]: https://bpb-us-e1.wpmucdn.com/sites.ucsc.edu/dist/7/1905/files/2025/03/erice.pdf

[^12]: https://core.ac.uk/download/pdf/73346567.pdf

[^13]: https://arxiv.org/pdf/math/0604082.pdf

[^14]: https://projecteuclid.org/journals/communications-in-mathematical-physics/volume-90/issue-3/The-order-parameter-in-a-spin-glass/cmp/1103940344.pdf

[^15]: https://journals.aps.org/pre/abstract/10.1103/PhysRevE.104.034102

[^16]: https://www.lewiscoleblog.com/spin-glass-models-2

[^17]: https://pubmed.ncbi.nlm.nih.gov/34021081/

[^18]: https://arxiv.org/abs/2102.11977

[^19]: https://www.ihes.fr/~ruelle/PUBLICATIONS/[90].pdf

[^20]: https://www.youtube.com/watch?v=i9EaYsR9cvA

[^21]: https://www.cirm-math.fr/RepOrga/2104/Slides/Baik.pdf

[^22]: https://link.aps.org/accepted/10.1103/PhysRevB.109.024431

[^23]: https://www.preprints.org/manuscript/202402.1058/v1

[^24]: https://hzshan.github.io/replica_method_in_SK_model.pdf

[^25]: https://www.math.toronto.edu/joaqsan/Resources/tomas_april2_slides.pdf

[^26]: https://indico.ictp.it/event/7607/session/348/contribution/1890/material/slides/0.pdf

[^27]: http://arxiv.org/pdf/1412.0170.pdf

[^28]: https://github.com/mcwitt/isg

[^29]: https://lewiscoleblog.com/spin-glass-models-4

[^30]: https://arxiv.org/pdf/1912.00793.pdf

[^31]: https://stackoverflow.com/questions/45399851/improving-python-code-in-monte-carlo-simulation/45403017

[^32]: https://ntrs.nasa.gov/api/citations/20140007519/downloads/20140007519.pdf

[^33]: https://github.com/mkan88/SGI

[^34]: https://developer.skao.int/projects/ska-sdp-resource-model/en/latest/_modules/ska_sdp_resource_model/simulate/monte_carlo.html

[^35]: https://www.g2qcomputing.com/QuantumMCMC.pdf

[^36]: https://sisl.readthedocs.io/en/v0.16.0/api/generated/sisl.physics.Overlap.html

[^37]: https://arxiv.org/abs/math/0604082

[^38]: https://arxiv.org/pdf/1905.03317.pdf

[^39]: https://pubs.aip.org/aip/jap/article/52/3/1697/503703/Faraday-rotation-measurements-of-time-dependent

[^40]: https://link.aps.org/accepted/10.1103/PhysRevB.96.054408

[^41]: https://www.numdam.org/item/AIHPB_2006__42_2_215_0/

[^42]: https://people.cas.uab.edu/~slstarr/lec1rev.pdf

[^43]: https://wt.iam.uni-bonn.de/fileadmin/WT/Inhalt/people/Anton_Bovier/lecture-notes/bovier-paris.pdf

[^44]: https://www.sciencedirect.com/science/article/abs/pii/0021999181900899

[^45]: https://web.stanford.edu/group/OTL/lagan/18012/SPINSUsageExample.pdf

[^46]: https://github.com/giselamarti/Exchange_MC

[^47]: https://www.jstor.org/stable/43736866

[^48]: http://arxiv.org/pdf/cond-mat/0609254.pdf

