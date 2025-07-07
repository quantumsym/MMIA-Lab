#!/usr/bin/python
#
import numpy as np
from plotly import *

def create_metropolis_plotly_animation():
    """
    Crea animazione interattiva di Metropolis-Hastings con Plotly
    """
    n_frames = 200
    sigma = 1.0
    def target_distribution(x):
        return np.exp(-x**2 / 2)
    # Simulazione completa
    samples = []
    x = 0.0
    acceptances = []
    for i in range(n_frames):
        x_new = x + np.random.normal(0, sigma)
        acceptance_ratio = min(1, target_distribution(x_new) / target_distribution(x))
        if np.random.random() < acceptance_ratio:
            x = x_new
            acceptances.append(True)
        else:
            acceptances.append(False)
        samples.append(x)
    # Crea frames per animazione
    frames = []
    for i in range(1, n_frames + 1):
        current_samples = samples[:i]
        frame = go.Frame(
            data=[
                go.Scatter(x=list(range(i)), y=current_samples, 
                          mode='lines+markers', name='Trace',
                          line=dict(color='blue')),
                go.Histogram(x=current_samples, nbinsx=20, 
                           name='Distribuzione', opacity=0.7,
                           yaxis='y2', histnorm='probability density')
            ],
            name=str(i)
        )
        frames.append(frame)
    # Figura iniziale
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Trace Plot Metropolis-Hastings', 'Distribuzione dei Campioni'),
        specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
    )
    # Trace plot iniziale
    fig.add_trace(
        go.Scatter(x=[^0], y=[samples[^0]], mode='lines+markers', name='Trace'),
        row=1, col=1
    )
    # Istogramma iniziale
    fig.add_trace(
        go.Histogram(x=[samples[^0]], nbinsx=20, name='Distribuzione', opacity=0.7),
        row=2, col=1
    )
    fig.frames = frames
    # Layout e controlli animazione
    fig.update_layout(
        title="Animazione Metropolis-Hastings",
        updatemenus=[{
            "buttons": [
                {
                    "args": [None, {"frame": {"duration": 100, "redraw": True},
                                   "fromcurrent": True}],
                    "label": "Play",
                    "method": "animate"
                },
                {
                    "args": [[None], {"frame": {"duration": 0, "redraw": True},
                                     "mode": "immediate", "transition": {"duration": 0}}],
                    "label": "Pause",
                    "method": "animate"
                }
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top"
        }],
        sliders=[{
            "active": 0,
            "steps": [{"args": [[f.name], {"frame": {"duration": 0, "redraw": True},
                                          "mode": "immediate"}],
                      "label": f.name, "method": "animate"}
                     for f in frames],
            "x": 0.1,
            "len": 0.9,
            "xanchor": "left",
            "y": 0,
            "yanchor": "top"
        }]
    )
    return fig

# Crea e mostra animazione (decommentare per eseguire)

fig_metro = create_metropolis_plotly_animation()
fig_metro.show()

