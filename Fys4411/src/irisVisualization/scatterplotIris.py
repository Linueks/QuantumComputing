#!/usr/bin/env python3
"""
Code for visualizing the data, found on 
https://stackoverflow.com/questions/45862223/use-different-colors-in-scatterplot-for-iris-dataset?newreg=a34c0972357041378fedf0e0302e2594
"""

import matplotlib.pyplot as plt
import seaborn as sns

iris = sns.load_dataset("iris")

iris["ID"] = iris.index

iris["ratio"] = iris["sepal_length"]/iris["sepal_width"]

sns.lmplot(x="ID", y="ratio", data=iris, hue="species", fit_reg=False, legend=False)

plt.legend()
plt.grid()
plt.show()

