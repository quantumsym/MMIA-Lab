#/usr/bin/env python
#
import numpy as np
import matplotlib.pyplot as plt

# Definizione dell'intervallo di x
x = np.linspace(-2, 2, 800)

# Calcolo della funzione
y = np.cos( 4 * np.pi  * x) * np.exp(-x**2)

# Creazione del grafico
plt.plot(x, y)
plt.show()

