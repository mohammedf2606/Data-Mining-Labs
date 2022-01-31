import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn import linear_model
from sklearn import metrics

from utils.utils import random_partition

x_rand, y_rand, p = datasets.make_regression(n_samples=100, n_features=1, n_informative=1, noise=10, coef=True)

trainset, testset = random_partition(x_rand, y_rand)

# actual plot

x_train, y_train = zip(*trainset)
x_test, y_test = zip(*testset)

x_input = [[float(1), float(x)] for x in x_train]

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
