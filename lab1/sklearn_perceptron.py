from sklearn import linear_model
from sklearn import datasets
from sklearn import metrics

from utils.utils import random_partition
import matplotlib.pyplot as plt

x, y = datasets.make_classification(n_features=1, n_redundant=0, n_informative=1, n_classes=2,
                                    n_clusters_per_class=1, n_samples=100)

trainset, testset = random_partition(x, y)

x_train, y_train = zip(*trainset)
x_test, y_test = zip(*testset)  # [(x,y) ...] -> [x], [y]

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
