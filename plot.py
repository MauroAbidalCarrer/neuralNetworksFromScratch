import matplotlib.pyplot as plt
import numpy as np

plt.ion()

# for probability classification
# def plot_samples(training_samples, test_samples, targets, bg_samples, bg_targets, X, Y):
#     plt.gca().cla()
#     colors = ['r', 'g', 'b']
#     # Assuming bg_targets are probabilities or scores, reshape them into a grid
#     bg_targets_grid = bg_targets.reshape(X.shape)
    
#     # Create a contour plot with a colorbar
#     contour = plt.contourf(X, Y, bg_targets_grid, cmap='RdYlBu', alpha=0.5)
    
#     # Draw training samples
#     for i in range(len(training_samples)):
#         plt.scatter(training_samples[i, 0], training_samples[i, 1], color=colors[targets[i]])
#     # Draw test samples
#     for i in range(len(test_samples)):
#         plt.scatter(test_samples[i, 0], test_samples[i, 1], color=colors[targets[i]], alpha=0.1)
    
#     plt.xlabel('X')
#     plt.ylabel('Y')
#     plt.draw()          # This causes the figure window to be updated with the current plot
#     plt.pause(0.001)    # This provides a short pause to allow the plot to be displayed


#     # Setting labels
#     plt.xlabel('X')
#     plt.ylabel('Y')

#     # Show the plot
#     plt.show()
#     plt.pause(1)

# For binary classification
# def plot_samples(training_samples, test_samples, targets, bg_targets, X, Y):
#     plt.gca().cla()
#     colors = ['r', 'g', 'b']
#     # Assuming bg_targets are probabilities or scores, reshape them into a grid
#     bg_targets_grid = bg_targets.reshape(X.shape)
    
#     # Create a contour plot with a colorbar
#     contour = plt.contourf(X, Y, bg_targets_grid, cmap='RdYlBu', alpha=0.5)
    
#     # Draw training samples
#     for i in range(len(training_samples)):
#         plt.scatter(training_samples[i, 0], training_samples[i, 1], color=colors[targets[i]])
#     # Draw test samples
#     for i in range(len(test_samples)):
#         plt.scatter(test_samples[i, 0], test_samples[i, 1], color=colors[targets[i]], alpha=0.1)
    
#     plt.xlabel('X')
#     plt.ylabel('Y')
#     plt.draw()          # This causes the figure window to be updated with the current plot
#     plt.pause(0.001)    # This provides a short pause to allow the plot to be displayed


#     # Setting labels
#     plt.xlabel('X')
#     plt.ylabel('Y')

#     # Show the plot
#     plt.show()
#     plt.pause(1)