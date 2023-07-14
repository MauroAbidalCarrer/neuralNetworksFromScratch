import matplotlib.pyplot as plt
import numpy as np

def plot_samples(samples, targets, bg_samples, bg_targets):
    # Define colors for classes and for the background
    colors = ['r', 'g', 'b']
    bg_colors = ['pink', 'lightgreen', 'lightblue']

    # Create a scatter plot for the background samples
    # print(bg_samples)
    print('len(bg_samples): ', len(bg_samples))
    print('bg_samples.shape: ', bg_samples.shape)
    print('bg_targets.shape: ', bg_targets.shape)
    for i in range(len(bg_samples)):
        # print('i: ', i, ', bg_targets[i]: ', bg_targets[i])
        bg_color = bg_colors[bg_targets[i]]
        plt.scatter(bg_samples[i, 0], bg_samples[i, 1], color=bg_color, marker='s', s=100)

    # Create a scatter plot for the actual data samples
    for i in range(len(samples)):
        plt.scatter(samples[i, 0], samples[i, 1], color=colors[targets[i]])

    # Setting labels
    plt.xlabel('X')
    plt.ylabel('Y')

    # Show the plot
    plt.show()
