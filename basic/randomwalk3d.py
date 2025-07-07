#!/usr/bin/python
#
from vpython import *

def vpython_monte_carlo_3d():
    '''
    Visualizzazione 3D di un random walk 3D usando VPython
    '''
    # Setup scena
    scene = canvas(title='3D Monte Carlo Random Walk',
                  width=800, height=600,
                  background=color.black)
    # Parametri
    n_steps = 1000
    step_size = 0.1
    # Posizione iniziale
    pos = vector(0, 0, 0)
    # Crea traccia del percorso
    trail = curve(color=color.cyan, radius=0.02)
    # Particella
    particle = sphere(pos=pos, radius=0.1, color=color.red)
    # Simulazione
    for i in range(n_steps):
        rate(50)  # 50 fps
        # Passo casuale 3D
        step = vector(
            random() - 0.5,
            random() - 0.5,
            random() - 0.5
        ) * step_size
        pos += step
        # Aggiorna posizione e traccia
        particle.pos = pos
        trail.append(pos)
        # Etichetta con statistiche
        if i % 100 == 0:
            distance = mag(pos)
            scene.caption = f'Step: {i}, Distance from origin: {distance:.2f}'

vpython_monte_carlo_3d()


