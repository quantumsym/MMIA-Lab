#!/usr/bin/env python
#
# brian_example.py
# This script demonstrates a simple neuron simulation using Brian2.
# It creates a single leaky integrate-and-fire neuron, injects a current,
# and records its membrane potential and spikes.

from brian2 import *
import matplotlib.pyplot as plt

# --- Simulation setup ---
# Set the default clock for the simulation.
start_scope()

# --- Neuron model definition ---
# Define the neuron model using differential equations.
# This is a leaky integrate-and-fire model.
tau = 10*ms
eqs = '''
dv/dt = (1-v)/tau : 1
'''

# --- Neuron and stimulus creation ---
# Create a neuron group with one neuron.
G = NeuronGroup(1, eqs, threshold='v>0.8', reset='v = 0', method='exact')

# --- Monitor setup ---
# Create a state monitor to record the membrane potential.
statemon = StateMonitor(G, 'v', record=0)

# Create a spike monitor to record spikes.
spikemon = SpikeMonitor(G)

# --- Applying stimulus ---
# Set the membrane potential of the neuron to a high value to make it spike.
G.v = 1.1

# --- Running the simulation ---
# Run the simulation for 100 milliseconds.
run(100 * ms)

# --- Plotting ---
plt.figure(figsize=(10, 6))

# Plot the membrane potential.
plt.plot(statemon.t/ms, statemon.v[0], label='Membrane Potential')

# Plot vertical lines for each spike.
for t in spikemon.t:
    plt.axvline(t/ms, ls='--', c='r', lw=2, label='Spike' if t == spikemon.t[0] else "")

# Add labels and title.
plt.xlabel('Time (ms)')
plt.ylabel('v')
plt.title('Brian2 Simulation: Single Leaky I&F Neuron')
plt.legend()
plt.grid(True)

# Save the plot to a file.
plt.savefig('brian_simulation.png')

print("Simulation finished. Plot saved to brian_simulation.png")



