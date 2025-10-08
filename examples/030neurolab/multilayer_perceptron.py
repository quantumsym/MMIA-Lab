#!/usr/bin/env python
import neurolab as nl

# Input: le quattro combinazioni di XOR
input_pattern = [[0,0], [0,1], [1,0], [1,1]]
target_pattern = [[0], [1], [1], [0]]

# Creazione della rete: 2 input, 1 hidden layer con 2 neuroni, 1 output
net = nl.net.newff([[0,1], [0,1]], [2, 1])

# Addestramento con backprop (breve, 100 epochs)
error = net.train(input_pattern, target_pattern, epochs=100, show=10, lr=0.1)

# Test
for inp in input_pattern:
    print(f'Input: {inp} -> Output: {net.sim([inp])[0][0]:.3f}')

