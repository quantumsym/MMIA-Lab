#!/usr/bin/env python
#

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def potential_energy(x):
    """
    Funzione di energia potenziale con molti minimi locali e un minimo globale centrale
    """
    # Funzione principale con minimo globale al centro
    main_potential =   (x - 1)**2 +  (x**2)*(0.75 * np.cos(np.pi * x ))**2

    return main_potential 

def create_static_plot():
    """
    Crea un grafico statico e lo salva in formato SVG
    """
    x_range = np.linspace(-6, 6, 1000)
    U = potential_energy(x_range)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(-6, 6)
    ax.set_ylim(np.min(U) - 2, np.max(U) + 2)
    ax.set_xlabel('Posizione x', fontsize=12)
    ax.set_ylabel('Energia Potenziale U(x)', fontsize=12)
    ax.set_title('Energia Potenziale con Minimi Locali e Globale', fontsize=14)
    ax.grid(True, alpha=0.3)

    # Plot della curva del potenziale
    ax.plot(x_range, U, 'b-', linewidth=3, label='Energia Potenziale')

    ax.legend()
    plt.tight_layout()

    # Salva in formato SVG
    plt.savefig('energia_potenziale.svg', format='svg', dpi=300, bbox_inches='tight')
    print("Grafico statico salvato come 'energia_potenziale.svg'")

    plt.show()

    return fig, ax

if __name__ == "__main__":
    # Crea il grafico statico e lo salva
    static_fig, static_ax = create_static_plot()


