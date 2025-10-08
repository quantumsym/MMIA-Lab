#!/usr/bin/python
#
import numpy as np
import matplotlib.pyplot as plt
N = 150  # number of steps


# random walk 1d
def random_walk_1d(N=100):
    # random number generator
    rng = np.random.default_rng()

    # generate N random equiprobable displacements, -1 or +1
    random_steps = rng.choice([-1, 1], size=N)

    # generate position vector with cumsum function (cumulative sum of an array)
    position = np.cumsum(random_steps)

    # insert initial position at the beginning of the list
    position = np.insert(position, 0, 0)  # add initial position

    return position

# visualization
def plot_random_walk(position, N=100):
    # plot of position after each step
    plt.plot(range(N + 1), position, marker='o')

    # titles, axis labels and grid
    plt.title('Random Walk 1D')
    plt.xlabel('Step')
    plt.ylabel('Position')
    plt.grid(True)

    # show the plot
    plt.show()


#
# Simulation execution
#
position = random_walk_1d(N)

# Calculate mean position and standard deviation
mean_position = np.mean(position)
std_position = np.std(position)

# Print statistics
print(f"Mean position: {mean_position:.4f}")
print(f"Standard deviation: {std_position:.4f}")
print(f"Final position: {position[-1]}")
print(f"Minimum position: {np.min(position)}")
print(f"Maximum position: {np.max(position)}")

# Display the plot
plot_random_walk(position,N)
