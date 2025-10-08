#!/usr/bin/env python
"""
Hopfield Recurrent Neural Network with CSV Data Loading
Task: Pattern Recognition (Letters or Binary Patterns)
"""
import numpy as np
import pandas as pd
import neurolab as nl

csv_filename = 'patterns.csv'  # target data file

def load_patterns_from_file(filename):
    # load data from file
    data = pd.read_csv(filename, header=None)  
    # Convert from binary (0,1) format  to bipolar (-1,1) format better for hopfield net
    patterns = data.to_numpy()   
    patterns = patterns.astype(float)
    patterns[patterns == 0] = -1
    return patterns

# load target patterns from file
target_patterns = load_patterns_from_file(csv_filename)

# neurolab.net.newhop() creates a Hopfield network and trains it using Hebbian learning rule
hopfield_net =  nl.net.newhop(target_patterns)

