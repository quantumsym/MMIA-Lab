#!/usr/bin/env python3
# numba_cpu_single.py
#
from numba import njit
import numpy as np
import matplotlib.pyplot as plt
import time

# ============================================================================
# PARAMETRI SIMULAZIONE
# ============================================================================
SIZE = 2048       # Dimensione immagine (pixels)
ITERATIONS = 1024    # Numero massimo di iterazioni
COLOUR_MAP = 'Greys'
# ============================================================================

@njit
def mandelbrot_njit(size, iterations):
    """Calculates the Mandelbrot set using Numba on a single CPU core."""
    image = np.zeros((size, size), dtype=np.int32)

    for i in range(size):
        for j in range(size):
            # Map pixel to complex plane
            c = complex(-2.0 + 3.0 * i / size, -1.5 + 3.0 * j / size)
            z = 0.0j

            for k in range(iterations):
                if z.real**2 + z.imag**2 >= 4.0:
                    break
                z = z**2 + c

            image[i, j] = k

    return image

def plot_mandelbrot(image, size, iterations,cmap):
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
    print("Generating Mandelbrot set...")
    print(f"Size: {SIZE}x{SIZE} pixels")
    print(f"Max iterations: {ITERATIONS}")
    print("-" * 50)

    # First run (includes JIT compilation time)
    print("First run (with JIT compilation)...")
    start_time = time.time()

    image = mandelbrot_njit(SIZE, ITERATIONS)

    first_run_time = time.time() - start_time
    print(f"Time: {first_run_time:.4f} seconds")

    # Second run (pure execution time)
    print("\nSecond run (pure execution)...")
    start_time = time.time()

    image = mandelbrot_njit(SIZE, ITERATIONS)

    second_run_time = time.time() - start_time
    print(f"Time: {second_run_time:.4f} seconds")

    print("-" * 50)
    print(f"JIT compilation overhead: {first_run_time - second_run_time:.4f} seconds")

    # Visualize the result
    print("\nDisplaying image...")
    plot_mandelbrot(image, SIZE, ITERATIONS,COLOUR_MAP)

if __name__ == "__main__":
    main()

