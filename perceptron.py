
#activation function

# Imagine you’re a light bulb, and someone claps their hands to turn you on. If the clap is loud enough, you light up. If it’s too quiet, you stay off.

# In your brain, activation functions do something similar for neurons in a computer. They listen to inputs (like claps) and decide if the neuron should “light up” (send a signal) or stay quiet.

# For example:

# ReLU (Rectified Linear Unit) is like: “If the clap is loud (a positive number), turn on the light. If it’s too quiet (negative), stay off.”

import matplotlib.pyplot as plt
import numpy as np
import random

class Perceptron:
  def __init__(self, num_inputs=3, weights=[1,1,1]):
    self.num_inputs = num_inputs
    self.weights = weights
    
  def weighted_sum(self, inputs):
    weighted_sum = 0
    for i in range(self.num_inputs):
      weighted_sum += self.weights[i]*inputs[i]
    return weighted_sum
  
  def activation(self, weighted_sum):
    if weighted_sum >= 0:
      return 1
    if weighted_sum < 0:
      return -1
    
  def training(self, training_set):
    foundLine = False
    while not foundLine:
      total_error = 0
      for inputs in training_set:
        prediction = self.activation(self.weighted_sum(inputs))
        actual = training_set[inputs]
        error = actual - prediction
        total_error += abs(error)
        for i in range(self.num_inputs):
          self.weights[i] += error*inputs[i]
        if total_error == 0:
          foundLine = True
      
# Visualization function
def plot_training_set(training_set, perceptron):
    # Plot training points
    for inputs, label in training_set.items():
        color = 'red' if label == 1 else 'blue'
        plt.scatter(inputs[0], inputs[1], color=color)

    # Plot decision boundary
    x = np.linspace(-5, 5, 100)
    y = - (perceptron.weights[0] / perceptron.weights[1]) * x
    plt.plot(x, y, '-g', label="Decision Boundary")
    plt.axhline(0, color='black', linewidth=0.5)  # x-axis
    plt.axvline(0, color='black', linewidth=0.5)  # y-axis
    plt.legend()
    plt.title("Perceptron Decision Boundary")
    plt.xlabel("Input 1")
    plt.ylabel("Input 2")
    plt.show()

# Training and visualization
cool_perceptron = Perceptron()
larger_training_set = {
    (0, 3, 1): 1,
    (3, 0, 1): -1,
    (0, -3, 1): -1,
    (3, -3, 1): 1,
    (-2, 2, 1): 1,
    (-3, -1, 1): -1,
    (2, -2, 1): -1,
    (4, 4, 1): 1,
    (-4, 5, 1): 1,
    (5, -4, 1): -1,
    (1, 1, 1): 1,
    (-1, -1, 1): -1,
    (2, 3, 1): 1,
    (-2, -3, 1): -1,
    (4, -1, 1): -1,
    (-4, 1, 1): 1,
    (0.5, 2.5, 1): 1,
    (-2.5, -1.5, 1): -1,
    (3.5, -2.5, 1): -1,
    (-1.5, 3.5, 1): 1
}


# Create a noisy version of the training set
noisy_training_set = {}
for inputs, label in larger_training_set.items():
    noisy_inputs = tuple(x + random.uniform(-0.5, 0.5) for x in inputs)  # Add small random noise
    noisy_training_set[noisy_inputs] = label

print("Noisy Training Set:", noisy_training_set)


# cool_perceptron.training(small_training_set)


# print("Final Weights:", cool_perceptron.weights)
# plot_training_set(small_training_set, cool_perceptron)

cool_perceptron = Perceptron()
cool_perceptron.training(noisy_training_set)
plot_training_set(noisy_training_set, cool_perceptron)
