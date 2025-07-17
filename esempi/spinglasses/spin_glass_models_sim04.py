
"""
system is slowly finding its way to a local energy minimum 
rather than exploring to find a better energy minima. 
To observe this better we will re-run the code for a second time 
from a different starting spot, we will compare the resulting spin array 

https://lewiscoleblog.com/spin-glass-models-2
"""

old_spins = spins
old_energy = energy[timesteps]
old_mag = mag[timesteps]

spins = np.random.choice([-1, 1], N)

#### Run Main Loop
_main_loop(timesteps, spins, interaction)

# Calculate a distance metric
dist = ((old_spins * spins).sum() / N + 1) / 2

print("Proportion of sites with the same spin is:", dist)
print("Resting energies of the 2 systems are:", old_energy, "and:", energy[timesteps])
print("Resting magnetism of the 2 systems are:", old_mag, "and:",  mag[timesteps])
