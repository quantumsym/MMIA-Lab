#!/usr/bin/env python3
import sympy as sp

x = sp.Symbol('x')
f = x**3 + 2*x**2 + 1
df = sp.diff(f, x)   # Derivata prima
print(df)            # Output: 3*x**2 + 4*x

# Derivata seconda
d2f = sp.diff(f, x, 2)
print(d2f)           # Output: 6*x + 4

# Valutazione numerica della derivata in x=1
print(df.subs(x, 1)) # Output: 7

