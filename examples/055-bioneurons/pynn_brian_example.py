#!/usr/bin/env python
# pynn_brian_example.py
# This script demonstrates a simple neuron simulation using PyNN with the Brian2 backend.
# It creates a single integrate-and-fire neuron, injects a constant current,
# and records its membrane potential and spikes.

import pyNN.brian2 as sim
import matplotlib.pyplot as plt

# --- Simulation setup ---
# Initialize the simulator. We use Brian2 as the backend.
# Set the simulation timestep to 0.1 ms.
sim.setup(timestep=0.1)

# --- Neuron and stimulus creation ---
# Create a population of one neuron of type IF_curr_alpha.
# This is a standard leaky integrate-and-fire model with alpha-shaped post-synaptic currents.
neuron_population = sim.Population(1, sim.IF_curr_alpha, {})

# Create a direct current (DC) source.
# This will inject a constant current into the neuron.
current_source = sim.DCSource(amplitude=0.5, start=20.0, stop=80.0)

# --- Connecting the components ---
# Connect the current source to the neuron population.
neuron_population.inject(current_source)


# --- Setting up recording ---
# Record the neuron's membrane potential (v) and spikes.
neuron_population.record(['v', 'spikes'])

# --- Running the simulation ---
# Run the simulation for 100 milliseconds.
sim.run(100.0)

# --- Data retrieval and plotting ---
# Get the recorded data from the neuron population.
data = neuron_population.get_data()

# Extract the membrane potential trace.
v_trace = data.segments[0].analogsignals[0]

# Extract the spike times.
spike_times = data.segments[0].spiketrains[0]

# Create a plot using matplotlib.
plt.figure(figsize=(10, 6))

# Plot the membrane potential.
plt.plot(v_trace.times, v_trace, label='Membrane Potential (mV)')

# Plot vertical lines for each spike.
for spike in spike_times:
    plt.axvline(spike, color='red', linestyle='--', lw=1, label='Spike' if spike == spike_times[0] else "")

# Add labels and title.
plt.xlabel('Time (ms)')
plt.ylabel('Membrane Potential (mV)')
plt.title('PyNN with Brian2 Backend: Single Leaky I&F Neuron')
plt.legend()
plt.grid(True)

# Save the plot to a file.
plt.savefig('pynn_brian_simulation.png')

print("Simulation finished. Plot saved to pynn_brian_simulation.png")

# --- End simulation ---
# Clean up the simulator resources.
sim.end()
