#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt


def make_numpy_arrays(filename, n_thetas, epochs=10):
    """
    As it stands this is a gross hardcoded filereader assuming:

    epochs = 10
    """
    with open("data\\"+filename+".dat") as fs:
        lines = fs.readlines()

    thetas_line = lines[1].strip('\\n').strip(' ')[1:-2]
    loss_line = lines[2].strip('\\n').strip(' ')[1:-2]
    accuracy_line = lines[3].strip('\\n').strip(' ')[1:-2]

    thetas = np.zeros(n_thetas)
    losses = np.zeros(10)
    accuracies = np.zeros(10)

    for i, n in enumerate(thetas_line.split()):
        thetas[i] = float(n)

    for i, n in enumerate(loss_line.split()):
        losses[i] = float(n)

    for i, n in enumerate(accuracy_line.split()):
        accuracies[i] = float(n)

    return thetas, losses, accuracies


if __name__=='__main__':
    #filename = "modelbasicModel_lrn0.5_shots1000_epochs10"
    parameterList = [5, 9, 5, 5, 9]
    modelList = ["basicModel", "doubleAnsatz", "lessEntangled", "doubleEncoding", "doubleAnsatzdoubleEncoding"]
    shotList = [1000, 10000]
    learnList = [0.1, 0.5, 1]
    epochs = 10

    """
    I want to plot each model in different plots
    I want different learning rates in the same plot
    I want different shots in different plots
    """

    for i, m in enumerate(modelList):
        for s in shotList:
            fig, ax = plt.subplots(figsize=(8, 5))
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            for l in learnList:
                filename = f'model{m}_lrn{l}_shots{s}_epochs{10}'
                thetas, losses, accuracies = make_numpy_arrays(filename, parameterList[i])
                ax.plot(losses, label=f'$\gamma = {l}$')
                ax2.plot(accuracies, label=f'$\gamma = {l}$')
                ax.set_title(f'Binary Cross Entropy for {m}')
                ax2.set_title(f'Prediction Accuracy for {m}')
                ax.set_xlabel('Epochs')
                ax2.set_xlabel('Epochs')
                ax.set_ylabel('$L(f_i)$')
                ax2.set_ylabel('Accuracy')
                ax.grid()
                ax2.grid()

                ax.legend()
                ax2.legend()

                fig.savefig(f'cross_entropy_model{str(m)}_shots{str(s)}_epochs{10}')
                fig2.savefig(f'accuracy_model{str(m)}_shots{str(s)}_epochs{10}')
    plt.show()
