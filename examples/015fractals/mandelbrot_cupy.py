#!/usr/bin/env python3
# mandelbrot_cupy.py
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import time

# ============================================================================
# PARAMETRI SIMULAZIONE
# ============================================================================
SIZE = 800          # Dimensione immagine (pixels)
ITERATIONS = 256    # Numero massimo di iterazioni
COLOURMAP = 'Greys' # Colormap per la visualizzazione
# ============================================================================

def mandelbrot_cupy(size, iterations):
    """Calcolo Mandelbrot con CuPy (GPU NVIDIA)."""
    # Crea la griglia nel piano complesso su GPU
    x = cp.linspace(-2.0, 1.0, size)
    y = cp.linspace(-1.5, 1.5, size)
    X, Y = cp.meshgrid(x, y, indexing='ij')
    C = X + 1j * Y

    # Inizializza Z e l'array dei contatori su GPU
    Z = cp.zeros_like(C, dtype=cp.complex128)
    K = cp.zeros(C.shape, dtype=cp.int32)

    # Iterazione di Mandelbrot
    for i in range(iterations):
        # Maschera per i punti che non sono ancora "esplosi"
        mask = cp.abs(Z) < 2.0
        # Aggiorna Z solo per i punti ancora validi
        Z = cp.where(mask, Z**2 + C, Z)
        # Aggiorna il contatore
        K = cp.where(mask, i, K)

    return K

def plot_mandelbrot(image, size, iterations, cmap):
    """Visualizza l'insieme di Mandelbrot."""
    plt.figure(figsize=(10, 10))
    plt.imshow(image.T, cmap=cmap, extent=[-2, 1, -1.5, 1.5],
               origin='lower', interpolation='bilinear')
    plt.colorbar(label='Iterations to escape')
    plt.title(f'Mandelbrot Set ({size}x{size}, {iterations} iterations)')
    plt.xlabel('Real axis')
    plt.ylabel('Imaginary axis')
    plt.tight_layout()
    plt.show()

def main():
    print("Generating Mandelbrot set with CuPy (NVIDIA GPU)...")
    print(f"Size: {SIZE}x{SIZE} pixels")
    print(f"Max iterations: {ITERATIONS}")
    print(f"GPU Device: {cp.cuda.Device().compute_capability}")
    print("-" * 50)

    # First run (includes data transfer overhead)
    print("First run (with GPU overhead)...")
    start_time = time.time()
    image_gpu = mandelbrot_cupy(SIZE, ITERATIONS)
    cp.cuda.Stream.null.synchronize()  # Sincronizza GPU
    first_run_time = time.time() - start_time
    print(f"Time: {first_run_time:.4f} seconds")

    # Second run (pure execution time)
    print("\nSecond run (warm cache)...")
    start_time = time.time()
    image_gpu = mandelbrot_cupy(SIZE, ITERATIONS)
    cp.cuda.Stream.null.synchronize()
    second_run_time = time.time() - start_time
    print(f"Time: {second_run_time:.4f} seconds")

    print("-" * 50)
    print(f"Overhead reduction: {first_run_time - second_run_time:.4f} seconds")

    # Copia il risultato da GPU a CPU per visualizzazione
    print("\nTransferring data from GPU to CPU...")
    image = np.array(image_gpu)

    # Visualize the result
    print("Displaying image...")
    plot_mandelbrot(image, SIZE, ITERATIONS, COLOURMAP)

if __name__ == "__main__":
    main()

