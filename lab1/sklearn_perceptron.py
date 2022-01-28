from sklearn import linear_model
from sklearn import datasets
from sklearn import metrics

import numpy as np
from utils.utils import random_partition
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
per.fit(x_train, y_train)
y_hat = per.predict(x_test)

print("accuracy = %f" % (metrics.accuracy_score(y_test, y_hat, normalize=True)))

x_train_class_1, y_train_class_1 = zip(*list(filter(lambda x: x[1] == 1, zip(x_train, y_train))))
x_train_class_0, y_train_class_0 = zip(*list(filter(lambda x: x[1] == 0, zip(x_train, y_train))))

x_test_class_1, y_test_class_1 = zip(*list(filter(lambda x: x[1] == 1, zip(x_test, y_test))))
x_test_class_0, y_test_class_0 = zip(*list(filter(lambda x: x[1] == 0, zip(x_test, y_test))))

plt.scatter(x_train_class_1, x_train_class_1, s=10, c='b', marker="o", label='train, class0')
plt.scatter(x_train_class_0, x_train_class_0, s=10, c='r', marker="o", label='train, class1')

plt.scatter(x_test_class_1, x_test_class_1, s=20, c='b', marker="x", label='test, class0')
plt.scatter(x_test_class_0, x_test_class_0, s=20, c='r', marker="x", label='test, class1')

y_plot = [per.intercept_ + j * per.coef_[0, 0] for j in x_train]

plt.plot(x_train, y_plot, color="black", label='slope')
plt.legend(loc='upper left')
plt.show()
