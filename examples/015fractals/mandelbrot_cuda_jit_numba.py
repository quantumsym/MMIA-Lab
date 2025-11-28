#!/usr/bin/env python3
# numba_cpu_single.py
#
from numba import cuda
import numpy as np
import matplotlib.pyplot as plt
import time

# ============================================================================
# PARAMETRI SIMULAZIONE
# ============================================================================
SIZE = 1600          # Dimensione immagine (pixels)
ITERATIONS = 512    # Numero massimo di iterazioni
COLOUR_MAP = 'Greys'
# ============================================================================

@cuda.jit
def mandelbrot_cuda(image, size, iterations):
    i, j = cuda.grid(2)
    if i < size and j < size:
        c_real = -2.0 + 3.0 * i / size
        c_imag = -1.5 + 3.0 * j / size
        c = complex(c_real, c_imag)
        z = 0.0j
        k = 0
        for n in range(iterations):
            if z.real**2 + z.imag**2 >= 4.0:
                break
            z = z**2 + c
            k = n
        image[i, j] = k

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

    # Nel main:
    image = np.zeros((SIZE, SIZE), dtype=np.int32)
    threadsperblock = (16, 16)
    blockspergrid_x = int(np.ceil(SIZE / threadsperblock[0]))
    blockspergrid_y = int(np.ceil(SIZE / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    
    d_image = cuda.to_device(image)
    mandelbrot_cuda[blockspergrid, threadsperblock](d_image, SIZE, ITERATIONS)
    d_image.copy_to_host(image)
    
    first_run_time = time.time() - start_time
    print(f"Time: {first_run_time:.4f} seconds")

    # Second run (pure execution time)
    print("\nSecond run (pure execution)...")
    start_time = time.time()

    # Nel main:
    image = np.zeros((SIZE, SIZE), dtype=np.int32)
    threadsperblock = (16, 16)
    blockspergrid_x = int(np.ceil(SIZE / threadsperblock[0]))
    blockspergrid_y = int(np.ceil(SIZE / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    
    d_image = cuda.to_device(image)
    mandelbrot_cuda[blockspergrid, threadsperblock](d_image, SIZE, ITERATIONS)
    d_image.copy_to_host(image)
    
    second_run_time = time.time() - start_time
    print(f"Time: {second_run_time:.4f} seconds")

    print("-" * 50)
    print(f"JIT compilation overhead: {first_run_time - second_run_time:.4f} seconds")

    # Visualize the result
    print("\nDisplaying image...")
    plot_mandelbrot(image, SIZE, ITERATIONS,COLOUR_MAP)

if __name__ == "__main__":
    main()

