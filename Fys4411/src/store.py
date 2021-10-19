#!/usr/bin/env python3

import numpy as np
"""
store into file:
name.dat [or something]
#first line with meta data, nr. shots, learning rate, ...
model theta values[..]
mean loss per epoch[..]
accuracy per epoch[..]
"""
filename = "somestring"
metaline = "meta data for 1st line"
dummyModel = np.array([1,3,45,6,3,2])
dummyLoss = [1,3,4,6,7,4]
dummyAccuracy = [43.2,4,2,5,6]

fs = open("data/"+filename+".dat","w")
fs.write(metaline); fs.write("\n")
fs.write(str(dummyModel)); fs.write("\n")
fs.write(str(dummyLoss)); fs.write("\n")
fs.write(str(dummyAccuracy));
fs.close()

