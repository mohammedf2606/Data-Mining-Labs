import matplotlib.pyplot as plt
from sklearn import datasets
import random as r

x_rand, y_rand, p = datasets.make_regression(n_samples=100, n_features=1, n_informative=1, noise=10, coef=True)

print(p)

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

plt.scatter(x_train, y_train, s=10, c='b', marker="s", label='first')
plt.scatter(x_test, y_test, s=10, c='r', marker="o", label='second')

plt.legend(loc='upper left')
plt.show()
