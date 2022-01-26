import matplotlib.pyplot as plt
import random as r
import numpy as np

from sklearn import datasets
from sklearn import linear_model
from sklearn import metrics


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

lr = linear_model.LinearRegression()
lr.fit(x_input, y_train)

predicted = lr.predict(x_input)

print("Regression equation: y = " + str(lr.intercept_) + " + " + str(lr.coef_[1]) + "x")
print(metrics.r2_score(y_train, predicted))
print(metrics.mean_squared_error(y_train, predicted))

plt.scatter(x_train, y_train, s=10, c='b', marker="s", label='first')
plt.plot(x_train, predicted, color="black")
plt.legend(loc='upper left')
plt.show()
