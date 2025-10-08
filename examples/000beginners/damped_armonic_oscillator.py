#!/usr/bin/env python
#
import numpy as np
import matplotlib.pyplot as plt

# Define the function for the damped harmonic oscillator
# y = sin(2πx) * exp(-x²)
# This represents a sinusoidal wave that decays exponentially with distance from the origin

# Create an array of x values from -10 to +10
# np.linspace creates evenly spaced values between start and stop
# 1000 points gives us a smooth curve for plotting
x = np.linspace(-2.5, 2.5, 1000)

# Calculate the function values
# np.sin() computes sine of each element in the array
# np.exp() computes exponential (e^x) of each element
# np.pi is the mathematical constant π ≈ 3.14159
y = np.sin(4 * np.pi * x) * np.exp(- x**2)

# Create the plot
# plt.figure() creates a new figure window
# figsize parameter sets the width and height in inches
plt.figure(figsize=(10, 6))

# Plot the function
# plt.plot() creates a line plot
# 'b-' means blue solid line
# linewidth controls the thickness of the line
# label is used for the legend
plt.plot(x, y, 'b-', linewidth=2, label=r'$y = \sin(2\pi x) \cdot e^{-x^2}$')

# Add grid for better readability
# alpha controls transparency (0=transparent, 1=opaque)
plt.grid(True, alpha=0.3)

# Add labels and title
# xlabel and ylabel add labels to the axes
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title('Damped Harmonic Oscillator: Symmetric Function', fontsize=14)

# Add legend to show the function equation
# loc='upper right' positions the legend in the upper right corner
plt.legend(loc='upper right')

# Set axis limits for better visualization
# xlim and ylim set the range of x and y axes displayed
plt.xlim(-2.5, 2.5)
plt.ylim(-1.1, 1.1)

# Add horizontal line at y=0 for reference
# 'k--' means black dashed line
# alpha makes it semi-transparent
plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)

# Display the plot
# plt.show() renders and displays the plot
plt.tight_layout()  # Adjusts spacing to prevent label cutoff
plt.show()

# Print some information about the function
print("Function: y = sin(2πx) * exp(-x²)")
print(f"Domain: [{x[0]:.1f}, {x[-1]:.1f}]")
print(f"Number of points calculated: {len(x)}")
print(f"Maximum value: {np.max(y):.4f}")
print(f"Minimum value: {np.min(y):.4f}")
