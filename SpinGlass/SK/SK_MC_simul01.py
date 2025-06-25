#!/usr/bin/python
#
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from scipy import stats
import SherringtonKirkpatrickMC

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

