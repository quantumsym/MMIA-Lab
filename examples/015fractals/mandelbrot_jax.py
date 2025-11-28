#!/usr/bin/env python3
# mandelbrot_jax.py
import matplotlib.pyplot as plt
import numpy as np
import time
import jax
import jax.numpy as jnp
import jax.lax as jl

# ============================================================================
# PARAMETRI SIMULAZIONE
# ============================================================================
SIZE = 2048          # Dimensione immagine (pixels)
ITERATIONS = 1024    # Numero massimo di iterazioni
COLOURMAP = 'Greys' # Colormap per la visualizzazione
# ============================================================================

def mandelbrot_jax_flexible(size, iterations):
    """Calcolo Mandelbrot con JAX usando fori_loop."""
    # Crea la griglia nel piano complesso
    x = jnp.linspace(-2.0, 1.0, size)
    y = jnp.linspace(-1.5, 1.5, size)
    X, Y = jnp.meshgrid(x, y, indexing='ij')
    C = X + 1j * Y

    def body_fun(i, val):
        Z, K = val
        mask = jnp.abs(Z) < 2.0
        Z = jnp.where(mask, Z**2 + C, Z)
        K = jnp.where(mask, i, K)
        return (Z, K)

    Z_init = jnp.zeros_like(C)
    K_init = jnp.zeros(C.shape, dtype=jnp.int32)
    _, K_final = jl.fori_loop(0, iterations, body_fun, (Z_init, K_init))

    return K_final

# JIT con static_argnames per SIZE
mandelbrot_jax_jit = jax.jit(mandelbrot_jax_flexible, static_argnames=['size'])

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
    print("Generating Mandelbrot set with JAX...")
    print(f"Size: {SIZE}x{SIZE} pixels")
    print(f"Max iterations: {ITERATIONS}")
    print("-" * 50)

    # First run (includes JIT compilation time)
    print("First run (with JIT compilation)...")
    start_time = time.time()
    image = np.array(mandelbrot_jax_jit(SIZE, ITERATIONS))
    first_run_time = time.time() - start_time
    print(f"Time: {first_run_time:.4f} seconds")

    # Second run (pure execution time)
    print("\nSecond run (pure execution)...")
    start_time = time.time()
    image = np.array(mandelbrot_jax_jit(SIZE, ITERATIONS))
    second_run_time = time.time() - start_time
    print(f"Time: {second_run_time:.4f} seconds")

    print("-" * 50)
    print(f"JIT compilation overhead: {first_run_time - second_run_time:.4f} seconds")

    # Visualize the result
    print("\nDisplaying image...")
    plot_mandelbrot(image, SIZE, ITERATIONS, COLOURMAP)

if __name__ == "__main__":
    main()

