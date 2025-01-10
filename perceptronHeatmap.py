from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt
import numpy as np
from itertools import product

data = [[0,0], [0,1], [1, 0],  [1,1]]
labels = [0, 0, 0, 1]

plt.scatter(
  [point[0] for point in data], [point[1] for point in data], c=labels
)


classifier = Perceptron(max_iter=40, random_state=22)

classifier.fit(data, labels)

print(classifier.score(data, labels))

print(classifier.decision_function([[0,0], [1,1], [0.5,0.5]]))

x_values = np.linspace(0, 1, 100)

y_values = np.linspace(0, 1, 100)

# print(list(x_values))
# print(list(y_values))

point_grid = list(product(x_values, y_values))


distances = classifier.decision_function(point_grid)

abs_distances = [abs(value) for value in distances]


# We’re almost ready to draw the heat map. We’re going to be using Matplotlib’s pcolormesh() function.

# Right now, abs_distances is a list of 10000 numbers. pcolormesh needs a two dimensional list. We need to turn abs_distances into a 100 by 100 two dimensional array.

# Numpy’s reshape function does this for us

distances_matrix = np.reshape(abs_distances, (100,100))
print(distances_matrix)

heatmap = plt.pcolormesh(x_values, y_values, distances_matrix)

plt.colorbar(heatmap)
plt.show()
