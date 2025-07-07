#!/usr/bin/env python3
from sympy import Symbol,diff,sin

x = Symbol('x')

# definizione della funzione
f = x**3 + 2*x**2 + sin(x)


print("Funzione: ",f)

df = diff(f, x)   # Derivata prima

print("Derivata prima: ",df)            # Output: 3*x**2 + 4*x

# Valutazione numerica della derivata prima in x=1
print("Valore per x=1 : ",df.subs(x, 1)) # Output: 7

# Derivata seconda
d2f = diff(f, x, 2)

print("Derivata seconda: ",d2f)           # Output: 6*x + 4


