#!/usr/bin/env python3

import numpy as np

filename = "modelbasicModel_lrn0.5_shots1000_epochs10"

with open("data\\"+filename+".dat") as fs:
    lines = fs.readlines()

n_parameters = [5, 9, 5, 9]

thetas_line = lines[1].strip('\\n').strip(' ')[1:-2]
loss_line = lines[2].strip('\\n').strip(' ')[1:-2]
accuracy_line = lines[3].strip('\\n').strip(' ')[1:-2]


print(thetas_line)
print(loss_line)
print(accuracy_line)

thetas = np.zeros(5)
losses = np.zeros(10)
accuracies = np.zeros(10)

for i, n in enumerate(thetas_line.split()):
    thetas[i] = float(n)

for i, n in enumerate(loss_line.split()):
    losses[i] = float(n)

for i, n in enumerate(accuracy_line.split()):
    accuracies[i] = float(n)

print(thetas)
print(losses)
print(accuracy)

def make_numpy_arrays(filename):
    """
    As it stands this is a gross hardcoded filereader assuming:

    epochs = 10
    """




"""
want to try variable
    models = [basic/naive, double theta, double encoding]
        shots = [1k, 10k]
            learningrates = [0.1, 0.5, 1]
"""
