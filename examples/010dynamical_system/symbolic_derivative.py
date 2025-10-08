#!/usr/bin/env python
import sympy as sp

x0 = 1.0

def fx():
    x = sp.symbols(var_name)
    expr_str = "x**3 + 2*x**2 + 1"
    expr = sp.sympify(expr_str)
    return expr_str

def f_compiled(expr, var_name='x'):
    f_compiled  = sp.lambdify(var_name, expr, 'numpy')
    return f_compiled

def symbolic_diff(expr, var_name='x'):
     x = sp.symbols(var_name)
     df_dx = sp.diff(expr, x)

     print(f"Function: {expr}")
     print(f"Symbolic derivative: {df_dx}")

     return  df_dx

df = symbolic_diff(fx)

df = f_compiled(df)

# numerical evalutations in x0

df_x0 = df.subs(x,x0)

print(df_x0)


