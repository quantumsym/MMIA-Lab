#!/usr/bin/env python
#
# Riemann integration estimate of pi
from numpy import sqrt,pi

# Set number of squares dividing the length of the [-1,1]x[-1,1] square
N = 2000

# Calculate square side length and square area
dx = 2 / N
da = dx**2

# Initiate estimate min and max
pi_min_est = 0
pi_max_est = 0

# Iterate over all possible squares
#
for x in range(1,N+1):
    for y in range(1,N+1):

        #top left corner distance from origin
        tl = sqrt(((x-1)*dx-1)**2+((y-1)*dx-1)**2)

        #top right corner distance from origin
        tr = sqrt(((x)*dx-1)**2+((y-1)*dx-1)**2)

        #bottom right corner distance from origin
        br = sqrt(((x)*dx-1)**2+((y)*dx-1)**2)

        #bottom right corner distance from origin
        bl = sqrt(((x-1)*dx-1)**2+((y)*dx-1)**2)

        # all corners lie within circle, add area to min  pi estimate
        if max(tl, tr, br, bl) <= 1:
            pi_min_est += da
        # one corner lie within circle, add area to max  pi estimate
        if min(tl, tr, br, bl) <= 1:
            pi_max_est += da

# mean between min and max estimation
pi_est = 0.5 * (pi_min_est + pi_max_est)

print("Estimate  of pi  :    ", pi_est)
print("Value of pi      :    ", pi)


