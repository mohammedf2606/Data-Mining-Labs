import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random as r

FILE_LOC = "./data.csv"

df = pd.read_csv(FILE_LOC, encoding='unicode_escape')
age_expec = df.iloc[0:, 70:72]

# x and y data cleaning

age_expec.drop(age_expec.loc[age_expec["Male life expectancy, (2012-14)"] == '.'].index, inplace=True)
age_expec.drop(age_expec.loc[age_expec["Female life expectancy, (2012-14)"] == '.'].index, inplace=True)


x = age_expec["Male life expectancy, (2012-14)"].values.tolist()[1:]
y = age_expec["Female life expectancy, (2012-14)"].values.tolist()[1:]
fig = plt.figure()
ax1 = fig.add_subplot(111)

train_size = int(len(x)*0.9)

all_indices = list(range(len(x)))
random_train_indices = r.sample(all_indices, train_size)
random_test_indices = list(set(all_indices) - set(random_train_indices))

print(random_train_indices)
print(random_test_indices)

index_values = list(enumerate(zip(x,y))) # Collection(index, (x,y))
trainset = [index_values[i][1] for i in random_train_indices] #(x,y)
testset = [index_values[i][1] for i in random_test_indices]

# print(train_size)
# print(len(testset))
# print(len(trainset))

# print(index_values)
# print(testset)
# print(trainset)

x_train, y_train = zip(*trainset)
x_test, y_test = zip(*testset)

# print("\n\n\n")
# print(x_test)
# print(y_test)

plt.scatter(x_train, y_train, s=10, c='b', marker="s", label='first')
plt.scatter(x_test, y_test, s=10, c='r', marker="o", label='second')

# plt.xlim([74, 88])
# ax1.set_ylim([78, 92])
plt.legend(loc='upper left')
plt.show()