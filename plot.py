import matplotlib.pyplot as plt
import numpy as np

def plot_samples(samples, targets):
    colors = ['r', 'g', 'b']  # r for class 0, g for class 1, b for class 2
    fig, ax = plt.subplots()
    for i in range(len(samples)):
        ax.scatter(samples[i, 0], samples[i, 1], color=colors[targets[i]])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.show()
