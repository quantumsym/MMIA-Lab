#!/usr/bin/python
def f(x):
    return x**2 + 3*x + 1

x = 2.0

h = 0.5 * e-5

# Derivata numerica (differenza centrata)
derivata = (f(x + h) - f(x - h)) / (2 * h)

print(derivata)  # Output: circa 7.0


