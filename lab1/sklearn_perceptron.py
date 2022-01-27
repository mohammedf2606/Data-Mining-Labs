from sklearn import linear_model
from sklearn import datasets
import random as r
import numpy as np

import matplotlib.pyplot as plt

x, y = datasets.make_classification(n_features=1, n_redundant=0, n_informative=1, n_classes=2,
                                    n_clusters_per_class=1, n_samples=100)

train_size = int(len(x) * 0.9)

all_indices = list(range(len(x)))
random_train_indices = r.sample(all_indices, train_size)
random_test_indices = list(set(all_indices) - set(random_train_indices))

x_unzip = list(zip(*x.tolist()))[0]

class0_train = [x_unzip[i] for i in range(len(list(zip(x_unzip, y))))
                for j in y if j == 0 and i in random_train_indices]
class1_train = [x_unzip[i] for i in range(len(list(zip(x_unzip, y))))
                for j in y if j == 1 and i in random_train_indices]

class0_test = [x_unzip[i] for i in range(len(list(zip(x_unzip, y))))
               for j in y if j == 0 and i in random_test_indices]
class1_test = [x_unzip[i] for i in range(len(list(zip(x_unzip, y))))
               for j in y if j == 1 and i in random_test_indices]

x_input = [[1, x] for x in class0_train]

print(class0_train)

per = linear_model.Perceptron()
per.fit(x_input, class0_train)
y_hat = per.predict(class0_train)

print("Regression equation: y = " + str(per.intercept_) + " + " + str(per.coef_[1]) + "x")

plt.scatter(class0_train, class0_train, s=10, c='b', marker="o", label='train, class0')
plt.scatter(class1_train, class1_train, s=10, c='r', marker="o", label='train, class1')

plt.scatter(class0_test, class0_test, s=10, c='b', marker="x", label='class0')
plt.scatter(class1_test, class1_test, s=10, c='r', marker="x", label='class1')

plt.plot(class0_train, y_hat, color="black")

plt.legend(loc='upper left')
plt.show()
