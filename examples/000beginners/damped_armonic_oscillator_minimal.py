#!/usr/bin/env python
#
import numpy as np
import matplotlib.pyplot as plt

# damped harmonic oscillator 
# a sinusoidal wave that decays exponentially with distance from the origin

x = np.linspace(-2.5, 2.5, 1000)
y = np.sin(4 * np.pi * x) * np.exp(- x**2)

plt.plot(x, y)
plt.show()

