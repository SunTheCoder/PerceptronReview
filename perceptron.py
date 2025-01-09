
#activation function

#   Imagine you’re a light bulb, and someone claps their hands to turn you on. If the clap is loud enough, you light up. If it’s too quiet, you stay off.

# In your brain, activation functions do something similar for neurons in a computer. They listen to inputs (like claps) and decide if the neuron should “light up” (send a signal) or stay quiet.

# For example:

# ReLU (Rectified Linear Unit) is like: “If the clap is loud (a positive number), turn on the light. If it’s too quiet (negative), stay off.”

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
      
cool_perceptron = Perceptron()
small_training_set = {(0,3,0):1, (3,0,0):-1, (0,-3,0):-1, (3,-3,0):1}
cool_perceptron.training(small_training_set)
print(cool_perceptron.weights) 