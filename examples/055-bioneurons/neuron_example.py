#!/usr/bin/env python
#
# neuron_example.py
# This script demonstrates a simple neuron simulation using NEURON.
# It creates a single soma, injects a current pulse,
# and records the membrane potential.

from neuron import h
import matplotlib.pyplot as plt

# --- Cell template ---
# Create a simple cell template with a single soma.
h.load_file("stdrun.hoc")

class MyCell:
    def __init__(self):
        self.soma = h.Section(name='soma', cell=self)
        self.soma.diam = 20
        self.soma.L = 20
        self.soma.insert('hh') # Hodgkin-Huxley kinetics

# --- Simulation setup ---
# Create an instance of the cell.
my_cell = MyCell()

# Create a current clamp.
stim = h.IClamp(my_cell.soma(0.5))
stim.delay = 10  # ms
stim.dur = 50    # ms
stim.amp = 0.1   # nA

# --- Recording setup ---
# Record time and voltage.
t_vec = h.Vector().record(h._ref_t)
v_vec = h.Vector().record(my_cell.soma(0.5)._ref_v)

# --- Simulation control ---
# Set the simulation parameters.
h.tstop = 100.0 # ms
h.run()

# --- Plotting ---
plt.figure(figsize=(10, 6))
plt.plot(t_vec, v_vec)
plt.xlabel("Time (ms)")
plt.ylabel("Membrane Potential (mV)")
plt.title("NEURON Simulation: Single Compartment HH Neuron")
plt.grid(True)
plt.savefig("neuron_simulation.png")

print("Simulation finished. Plot saved to neuron_simulation.png")
