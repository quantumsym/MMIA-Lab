
#!/usr/bin/env python3
# mandelbrot_with_decorator.py
import time
import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange

# ============================================================================
# PARAMETRI SIMULAZIONE
# ============================================================================
SIZE = 800          # Dimensione immagine (pixels)
ITERATIONS = 256    # Numero massimo di iterazioni
COLOURMAP = 'Greys' # Colormap per la visualizzazione
# ============================================================================

# Decoratore per misurare il tempo
def timer(func):
    """Decoratore per misurare il tempo di esecuzione."""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        elapsed = end - start
        print(f"  ⏱️  {func.__name__}: {elapsed:.4f}s")
        return result
    return wrapper

# ============================================================================
# VERSIONE 1: NumPy Puro
# ============================================================================
@timer
def mandelbrot_numpy(size, iterations):
    """Calcolo Mandelbrot puro in NumPy (loop in Python)."""
    image = np.zeros((size, size), dtype=np.int32)
    for i in range(size):
        for j in range(size):
            c = complex(-2.0 + 3.0 * i / size, -1.5 + 3.0 * j / size)
            z = 0.0j
            for k in range(iterations):
                if z.real**2 + z.imag**2 >= 4.0:
                    break
                z = z**2 + c
            image[i, j] = k
    return image

# ============================================================================
# VERSIONE 2: Numba @njit
# ============================================================================
@njit
def mandelbrot_numba_core(size, iterations):
    """Kernel Mandelbrot puro Numba."""
    image = np.zeros((size, size), dtype=np.int32)
    for i in range(size):
        for j in range(size):
            c = complex(-2.0 + 3.0 * i / size, -1.5 + 3.0 * j / size)
            z = 0.0j
            for k in range(iterations):
                if z.real**2 + z.imag**2 >= 4.0:
                    break
                z = z**2 + c
            image[i, j] = k
    return image

@timer
def mandelbrot_numba(size, iterations):
    """Wrapper con timer per Numba."""
    return mandelbrot_numba_core(size, iterations)

# ============================================================================
# VERSIONE 3: Numba Parallelo (prange)
# ============================================================================
@njit(parallel=True, fastmath=True, cache=True)
def mandelbrot_numba_parallel_core(size, iterations):
    """Kernel Mandelbrot parallelo con prange."""
    image = np.zeros((size, size), dtype=np.int32)
    for i in prange(size):
        for j in range(size):
            c = complex(-2.0 + 3.0 * i / size, -1.5 + 3.0 * j / size)
            z = 0.0j
            for k in range(iterations):
                if z.real**2 + z.imag**2 >= 4.0:
                    break
                z = z**2 + c
            image[i, j] = k
    return image

@timer
def mandelbrot_numba_parallel(size, iterations):
    """Wrapper con timer per Numba Parallelo."""
    return mandelbrot_numba_parallel_core(size, iterations)

# ============================================================================
# VERSIONE 4: JAX
# ============================================================================
try:
    import jax
    import jax.numpy as jnp

    def mandelbrot_jax_core(size, iterations):
        """Kernel Mandelbrot JAX."""
        x = jnp.linspace(-2.0, 1.0, size)
        y = jnp.linspace(-1.5, 1.5, size)
        X, Y = jnp.meshgrid(x, y, indexing='ij')
        C = X + 1j * Y
        Z = jnp.zeros_like(C)
        K = jnp.zeros(C.shape, dtype=jnp.int32)
        for i in range(iterations):
            mask = jnp.abs(Z) < 2.0
            Z = jnp.where(mask, Z**2 + C, Z)
            K = jnp.where(mask, i, K)
        return K

    mandelbrot_jax_jit = jax.jit(mandelbrot_jax_core, static_argnames=['size'])

    @timer
    def mandelbrot_jax(size, iterations):
        """Wrapper con timer per JAX."""
        return np.array(mandelbrot_jax_jit(size, iterations))

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    print("⚠️  JAX non disponibile")

# ============================================================================
# VERSIONE 5: CuPy (se disponibile)
# ============================================================================
try:
    import cupy as cp

    @timer
    def mandelbrot_cupy(size, iterations):
        """Calcolo Mandelbrot con CuPy (GPU NVIDIA)."""
        x = cp.linspace(-2.0, 1.0, size)
        y = cp.linspace(-1.5, 1.5, size)
        X, Y = cp.meshgrid(x, y, indexing='ij')
        C = X + 1j * Y
        Z = cp.zeros_like(C, dtype=cp.complex128)
        K = cp.zeros(C.shape, dtype=cp.int32)
        for i in range(iterations):
            mask = cp.abs(Z) < 2.0
            Z = cp.where(mask, Z**2 + C, Z)
            K = cp.where(mask, i, K)
        cp.cuda.Stream.null.synchronize()
        return np.array(K)

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    print("⚠️  CuPy non disponibile")

# ============================================================================
# VISUALIZZAZIONE
# ============================================================================
def plot_mandelbrot(image, size, iterations, cmap, title_suffix=""):
    """Visualizza l'insieme di Mandelbrot."""
    plt.figure(figsize=(10, 10))
    plt.imshow(image.T, cmap=cmap, extent=[-2, 1, -1.5, 1.5],
               origin='lower', interpolation='bilinear')
    plt.colorbar(label='Iterations to escape')
    plt.title(f'Mandelbrot Set ({size}x{size}, {iterations} it.) {title_suffix}')
    plt.xlabel('Real axis')
    plt.ylabel('Imaginary axis')
    plt.tight_layout()
    plt.show()

# ============================================================================
# MAIN: BENCHMARK COMPARATIVO
# ============================================================================
def main():
    print("\n" + "="*60)
    print("BENCHMARK MANDELBROT - Confronto Implementazioni")
    print("="*60)
    print(f"Parametri: SIZE={SIZE}, ITERATIONS={ITERATIONS}\n")

    results = {}

    # Test 1: NumPy puro (lento!)
    print("1️⃣  NumPy Puro")
    try:
        result_numpy = mandelbrot_numpy(SIZE, ITERATIONS)
        results['NumPy'] = result_numpy
    except Exception as e:
        print(f"  ❌ Errore: {e}")

    # Test 2: Numba @njit
    print("\n2️⃣  Numba @njit (single-core)")
    print("  (Prima esecuzione include compilazione JIT)")
    try:
        result_numba = mandelbrot_numba(SIZE, ITERATIONS)
        results['Numba'] = result_numba
    except Exception as e:
        print(f"  ❌ Errore: {e}")

    # Test 3: Numba Parallelo
    print("\n3️⃣  Numba Parallelo (@njit parallel=True)")
    try:
        result_numba_par = mandelbrot_numba_parallel(SIZE, ITERATIONS)
        results['Numba Parallelo'] = result_numba_par
    except Exception as e:
        print(f"  ❌ Errore: {e}")

    # Test 4: JAX
    if JAX_AVAILABLE:
        print("\n4️⃣  JAX (@jit)")
        try:
            result_jax = mandelbrot_jax(SIZE, ITERATIONS)
            results['JAX'] = result_jax
        except Exception as e:
            print(f"  ❌ Errore: {e}")

    # Test 5: CuPy
    if CUPY_AVAILABLE:
        print("\n5️⃣  CuPy (GPU NVIDIA)")
        try:
            result_cupy = mandelbrot_cupy(SIZE, ITERATIONS)
            results['CuPy'] = result_cupy
        except Exception as e:
            print(f"  ❌ Errore: {e}")

    print("\n" + "="*60)
    print("Visualizzazione del risultato...")
    print("="*60 + "\n")

    # Mostra il primo risultato disponibile
    if results:
        first_result = list(results.values())[0]
        first_name = list(results.keys())[0]
        plot_mandelbrot(first_result, SIZE, ITERATIONS, COLOURMAP, f"({first_name})")

if __name__ == "__main__":
    main()

