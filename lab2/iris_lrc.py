import numpy as np
import sklearn.datasets as data
import sklearn.model_selection as model_select
import sklearn.linear_model as linear_model
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from sklearn import metrics

iris = data.load_iris()

x = iris.data[:, :2]
y = iris.target

x_train, x_test, y_train, y_test = model_select.train_test_split(x[:, :2], y, random_state=0)

clf = linear_model.LogisticRegression(solver='lbfgs', multi_class='multinomial')
clf.fit(x_train, y_train)

y_hat = clf.predict(x_test)

correct_predictions = [i == j for i, j in zip(y_hat, y_test)]
score = (sum(correct_predictions) / len(x_test)) * 100
print("number of correct predictions = %d out of %d = %f%%" % (sum(correct_predictions), len(x_test), score))

print("training score = ", clf.score(x_train, y_train))
print("test score = ", clf.score(x_test, y_test))

print("accuracy score = ", metrics.accuracy_score(y_test, y_hat))

confusion_matrix = metrics.confusion_matrix(y_test, y_hat)

confusion_matrix_table = PrettyTable()
confusion_matrix_table.field_names = iris.target_names
for el in confusion_matrix:
    confusion_matrix_table.add_row(el)
print(confusion_matrix_table)

print('\nPrecision Score:')
precision = metrics.precision_score(y_test, y_hat, average=None)
for i, j in zip(iris.target_names, precision):
    print(i + " = " + str(j))

print('\nRecall Score:')
recall = metrics.recall_score(y_test, y_hat, average=None)
for i, j in zip(iris.target_names, recall):
    print(i + " = " + str(j))

print('\nf1 Score:')
f1 = metrics.f1_score(y_test, y_hat, average=None)
for i, j in zip(iris.target_names, recall):
    print(i + " = " + str(j))

sepal_length = [i[0] for i in x]
sepal_width = [i[1] for i in x]

min_max_table = PrettyTable()
min_max_table.field_names = ["", "min", "max"]
min_max_table.add_row(["sepal length", min(sepal_length), max(sepal_length)])
min_max_table.add_row(["sepal width", min(sepal_width), max(sepal_width)])

print(min_max_table)

x_pairs = []
GRANULARITY = 0.1
x0_range = np.arange(min(sepal_length), max(sepal_length), GRANULARITY)
x1_range = np.arange(min(sepal_width), max(sepal_width), GRANULARITY)
for i, j in zip(x0_range, x1_range):
    x_pairs.append((round(i, 2), round(j, 2)))
y_hat_pairs = clf.predict(x_pairs)
print("mesh score = ", clf.score(x_pairs, y_hat_pairs))
x0_mesh, x1_mesh = np.meshgrid(x0_range, x1_range)
y_hat_mesh = y_hat_pairs.reshape(x0_mesh.shape)
plt.pcolormesh(x0_mesh, x1_mesh, y_hat_mesh, shading="flat")
plt.set_cmap("Blues")
plt.plot()
