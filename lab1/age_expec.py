import pandas as pd
import matplotlib.pyplot as plt
import random as r
import numpy as np

from utils import gradient_descent as gd
from utils.utils import compute_r2

FILE_LOC = "./data.csv"

df = pd.read_csv(FILE_LOC, encoding='unicode_escape')
age_expec = df.iloc[0:, 70:72]

# x and y data cleaning

age_expec.drop(age_expec.loc[age_expec["Male life expectancy, (2012-14)" or
                                       "Female life expectancy, (2012-14)"] == '.'].index, inplace=True)
age_expec = age_expec.dropna()

x = age_expec["Male life expectancy, (2012-14)"]
x = pd.to_numeric(x, errors='coerce').values.tolist()  # Force all values to float

y = age_expec["Female life expectancy, (2012-14)"]
y = pd.to_numeric(y, errors='coerce').values.tolist()  # Force all values to float

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

x_input = [[float(1), float(x)] for x in x_train]
weights = [10 for i in range(len(x_input[0]))]
LEARNING_RATE = 0.0001
predicted = []

for _ in range(1000):
    weights, predicted = gd.batch_gradient_descent(len(x_input), x_input, weights, y_train, LEARNING_RATE)

print(compute_r2(predicted, y_train))

plt.scatter(x_train, y_train, s=10, c='b', marker="s", label='first')

ax = plt.gca()
X_plot = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 100)
plt.plot(X_plot, weights[1]*X_plot + weights[0], color='C0', label='by slope')
plt.show()
