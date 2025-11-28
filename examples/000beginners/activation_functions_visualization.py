#!/usr/bin/env python3
# activation_functions_visualization.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ============================================================================
# PARAMETRI
# ============================================================================
X_MIN = -5.0
X_MAX = 5.0
POINTS = 1000
OUTPUT_DIR = "./activation_functions_svg/"
# ============================================================================

# Genera i dati di input
x = np.linspace(X_MIN, X_MAX, POINTS)

# ============================================================================
# FUNZIONI DI ATTIVAZIONE
# ============================================================================

def sigmoid(x):
    """Sigmoid: 1 / (1 + e^(-x))"""
    return 1.0 / (1.0 + np.exp(-x))

def tanh_act(x):
    """Tanh: (e^x - e^(-x)) / (e^x + e^(-x))"""
    return np.tanh(x)

def relu(x):
    """ReLU (Rectified Linear Unit): max(0, x)"""
    return np.maximum(0, x)

def step(x):
    """Heaviside Step Function: 0 se x < 0, 1 se x >= 0"""
    return np.where(x >= 0, 1.0, 0.0)

def linear_saturated(x, a=-2.0, b=2.0):
    """Linear Saturation: clip(x, a, b)"""
    return np.clip(x, a, b)

def softmax_1d(x):
    """Softmax 1D: per un singolo input, si normalizza come e^x / sum(e^x)"""
    # Per visualizzazione, mostriamo il softmax su un vettore di valori
    # Simuliamo softmax con 5 classi
    return np.exp(x) / np.sum(np.exp(x))

# ============================================================================
# VISUALIZZAZIONE SINGOLE FUNZIONI
# ============================================================================

def plot_single_activation(x, y, title, filename, color='#2E86AB'):
    """Crea e salva un singolo grafico di attivazione."""
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(x, y, linewidth=2.5, color=color, label=title)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.set_xlabel('Input (x)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Output', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.set_xlim(X_MIN, X_MAX)

    plt.tight_layout()
    plt.savefig(filename, format='svg', dpi=300)
    plt.close()
    print(f"✓ Salvato: {filename}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    import os

    # Crea directory se non esiste
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n" + "="*60)
    print("VISUALIZZAZIONE FUNZIONI DI ATTIVAZIONE")
    print("="*60 + "\n")

    # 1. Sigmoid
    print("1. Elaborando Sigmoid...")
    y_sigmoid = sigmoid(x)
    plot_single_activation(x, y_sigmoid, "Sigmoid Function",
                          f"{OUTPUT_DIR}01_sigmoid.svg", color='#E63946')

    # 2. Tanh
    print("2. Elaborando Tanh...")
    y_tanh = tanh_act(x)
    plot_single_activation(x, y_tanh, "Tanh (Hyperbolic Tangent)",
                          f"{OUTPUT_DIR}02_tanh.svg", color='#F77F00')

    # 3. ReLU
    print("3. Elaborando ReLU...")
    y_relu = relu(x)
    plot_single_activation(x, y_relu, "ReLU (Rectified Linear Unit)",
                          f"{OUTPUT_DIR}03_relu.svg", color='#06A77D')

    # 4. Step Function
    print("4. Elaborando Step Function...")
    y_step = step(x)
    plot_single_activation(x, y_step, "Heaviside Step Function",
                          f"{OUTPUT_DIR}04_step.svg", color='#8E44AD')

    # 5. Linear Saturation
    print("5. Elaborando Linear Saturation...")
    y_saturated = linear_saturated(x)
    plot_single_activation(x, y_saturated, "Linear Saturation (Clipped)",
                          f"{OUTPUT_DIR}05_linear_saturated.svg", color='#D62828')

    # 6. Softmax (visualizzazione speciale)
    print("6. Elaborando Softmax...")
    fig, ax = plt.subplots(figsize=(8, 6))

    # Softmax con 5 classi simulate
    classes = np.arange(5)
    softmax_output = softmax_1d(np.array([0.5, 1.5, -0.5, 2.0, -1.0]))

    # Per una visualizzazione continua, mostriamo softmax su una griglia
    x_softmax = np.linspace(X_MIN, X_MAX, POINTS)
    # Simuliamo softmax con bias fissi
    biases = np.array([-1, -0.5, 0, 0.5, 1])
    softmax_curves = []
    for bias in biases:
        exp_vals = np.exp(x_softmax + bias)
        exp_sum = np.exp(x_softmax - 1) + np.exp(x_softmax - 0.5) + \
                  np.exp(x_softmax) + np.exp(x_softmax + 0.5) + np.exp(x_softmax + 1)
        softmax_curves.append(exp_vals / exp_sum)

    colors_softmax = ['#E63946', '#F77F00', '#06A77D', '#2E86AB', '#8E44AD']
    for i, curve in enumerate(softmax_curves):
        ax.plot(x_softmax, curve, linewidth=2.0, label=f'Class {i}',
                color=colors_softmax[i], alpha=0.8)

    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlabel('Input (x)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Probability', fontsize=12, fontweight='bold')
    ax.set_title('Softmax Function (5 Classes)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}06_softmax.svg", format='svg', dpi=300)
    plt.close()
    print(f"✓ Salvato: {OUTPUT_DIR}06_softmax.svg")

    # ========================================================================
    # GRAFICO COMPARATIVO
    # ========================================================================
    print("7. Creando grafico comparativo...")

    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

    # Subplot 1: Sigmoid
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(x, sigmoid(x), linewidth=2.5, color='#E63946')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linewidth=0.8)
    ax1.axvline(x=0, color='black', linewidth=0.8)
    ax1.set_title('Sigmoid', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Output', fontsize=10)

    # Subplot 2: Tanh
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(x, tanh_act(x), linewidth=2.5, color='#F77F00')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linewidth=0.8)
    ax2.axvline(x=0, color='black', linewidth=0.8)
    ax2.set_title('Tanh', fontsize=12, fontweight='bold')

    # Subplot 3: ReLU
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(x, relu(x), linewidth=2.5, color='#06A77D')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='black', linewidth=0.8)
    ax3.axvline(x=0, color='black', linewidth=0.8)
    ax3.set_title('ReLU', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Output', fontsize=10)

    # Subplot 4: Step Function
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(x, step(x), linewidth=2.5, color='#8E44AD', drawstyle='steps-mid')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='black', linewidth=0.8)
    ax4.axvline(x=0, color='black', linewidth=0.8)
    ax4.set_title('Step Function', fontsize=12, fontweight='bold')
    ax4.set_ylim(-0.1, 1.1)

    # Subplot 5: Linear Saturation
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(x, linear_saturated(x), linewidth=2.5, color='#D62828')
    ax5.grid(True, alpha=0.3)
    ax5.axhline(y=0, color='black', linewidth=0.8)
    ax5.axvline(x=0, color='black', linewidth=0.8)
    ax5.set_title('Linear Saturation', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Input (x)', fontsize=10)
    ax5.set_ylabel('Output', fontsize=10)

    # Subplot 6: Comparazione Sigmoid vs Tanh
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.plot(x, sigmoid(x), linewidth=2.5, label='Sigmoid', color='#E63946', alpha=0.8)
    ax6.plot(x, tanh_act(x), linewidth=2.5, label='Tanh', color='#F77F00', alpha=0.8)
    ax6.plot(x, relu(x), linewidth=2.5, label='ReLU', color='#06A77D', alpha=0.8)
    ax6.grid(True, alpha=0.3)
    ax6.axhline(y=0, color='black', linewidth=0.8)
    ax6.axvline(x=0, color='black', linewidth=0.8)
    ax6.set_title('Confronto Funzioni', fontsize=12, fontweight='bold')
    ax6.legend(fontsize=10, loc='best')
    ax6.set_xlabel('Input (x)', fontsize=10)

    plt.suptitle('Funzioni di Attivazione Neurali', fontsize=16, fontweight='bold', y=0.995)
    plt.savefig(f"{OUTPUT_DIR}00_comparison.svg", format='svg', dpi=300)
    plt.close()
    print(f"✓ Salvato: {OUTPUT_DIR}00_comparison.svg")

    print("\n" + "="*60)
    print("✅ COMPLETATO! Tutti i file SVG salvati in: " + OUTPUT_DIR)
    print("="*60 + "\n")

if __name__ == "__main__":
    main()

