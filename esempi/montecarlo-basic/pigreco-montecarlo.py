#!/usr/bin/env python
# Monte-Carlo estimate of pi
from numpy.random import random
from numpy import sqrt,pi

# Set number of simulated points
N = 100000

# Initiate counter
count = 0

# Generate random points within [1,1]x[-1,1] square and test
# the random(2) function extracts a pair of decimal numbers between [0,0] e [1,1]
# the product by 2 minus 1 scales them to fall within  [-1,1]x[-1,1]
#  random(n) extracts an n-tuple of decimal numbers between 0 and 1

for i in range(N):
    x,y  = 2* random(2) - 1

    # add 4 to the score if the point falls within the unitary circle
    if sqrt( x**2 + y**2) < 1:
        count += 4    

print("Estimate of pi:", count/N)
print("Value    of pi:", pi)



