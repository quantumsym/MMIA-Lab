
import neurolab as nl

# Logical &
input = [[0, 0], [0, 1], [1, 0], [1, 1]]
target = [[0], [0], [0], [1]]

# Create net with 2 inputs and 1 neuron
net = nl.net.newp([[0, 1],[0, 1]], 1)

# train with delta rule to classify with AND logical function
error = net.train(input, target, epochs=100, show=10, lr=0.1)


