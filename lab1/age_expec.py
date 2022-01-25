import pandas as pd
import matplotlib.pyplot as plt
import random as r

from utils import gradient_descent as gd

FILE_LOC = "./data.csv"

df = pd.read_csv(FILE_LOC, encoding='unicode_escape')
age_expec = df.iloc[0:, 70:72]

# x and y data cleaning

age_expec.drop(age_expec.loc[age_expec["Male life expectancy, (2012-14)"] == '.'].index, inplace=True)
age_expec.drop(age_expec.loc[age_expec["Female life expectancy, (2012-14)"] == '.'].index, inplace=True)
age_expec = age_expec.dropna()

x = age_expec["Male life expectancy, (2012-14)"].values.tolist()
y = age_expec["Female life expectancy, (2012-14)"].values.tolist()
fig = plt.figure()
ax1 = fig.add_subplot(111)

train_size = int(len(x)*0.9)

all_indices = list(range(len(x)))
random_train_indices = r.sample(all_indices, train_size)
random_test_indices = list(set(all_indices) - set(random_train_indices))

index_values = list(enumerate(zip(x, y)))  # Collection(index, (x,y))
trainset = [index_values[i][1] for i in random_train_indices]  # (x,y)
testset = [index_values[i][1] for i in random_test_indices]

# actual plot

x_train, y_train = zip(*trainset)
x_test, y_test = zip(*testset)

# plt.scatter(x_train, y_train, s=10, c='b', marker="s", label='first')
# plt.scatter(x_test, y_test, s=10, c='r', marker="o", label='second')
#
# plt.legend(loc='upper left')
# plt.show()
x_input = [[float(1), float(x)] for x in x_train]
weights = [1 for i in range(len(x_input[0]))]
LEARNING_RATE = 0.001

for _ in range(10):
    print(weights)
    weights = gd.batch_gradient_descent(len(x_input), x_input, weights, y_train, LEARNING_RATE)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x_train, y_train, s=10, c='b', marker="s", label='first')
    ax.axline((0, weights[0]), slope=weights[1], color='C0', label='by slope')
    plt.show()


