import matplotlib.pyplot as plt
from sklearn import datasets
import random as r
import numpy as np

from utils import gradient_descent as gd
from utils.utils import compute_r2

x_rand, y_rand, p = datasets.make_regression(n_samples=100, n_features=1, n_informative=1, noise=10, coef=True)

train_size = int(len(x_rand)*0.9)

all_indices = list(range(len(x_rand)))
random_train_indices = r.sample(all_indices, train_size)
random_test_indices = list(set(all_indices) - set(random_train_indices))

index_values = list(enumerate(zip(x_rand, y_rand)))  # Collection(index, (x,y))
trainset = [index_values[i][1] for i in random_train_indices]  # (x,y)
testset = [index_values[i][1] for i in random_test_indices]

# actual plot

x_train, y_train = zip(*trainset)
x_test, y_test = zip(*testset)

x_input = [[float(1), float(x)] for x in x_train]
weights = [1 for i in range(len(x_input[0]))]
LEARNING_RATE = 0.0001
predicted = []

for _ in range(10000):
    weights, predicted = gd.batch_gradient_descent(len(x_input), x_input, weights, y_train, LEARNING_RATE)

print(compute_r2(predicted, y_train))

plt.scatter(x_train, y_train, s=10, c='b', marker="s", label='first')

ax = plt.gca()
X_plot = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 100)
plt.plot(X_plot, weights[1]*X_plot + weights[0], color='C0', label='by slope')
plt.show()
