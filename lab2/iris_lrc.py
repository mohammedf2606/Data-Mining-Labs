import numpy as np
import sklearn.datasets as data
import sklearn.model_selection as model_select
import sklearn.linear_model as linear_model
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from sklearn import metrics
import sklearn.preprocessing as preprocess

iris = data.load_iris()

x = iris.data[:, :2]
y = iris.target

x_train, x_test, y_train, y_test = model_select.train_test_split(x, y, random_state=0)

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

GRANULARITY = 0.01
x0_min = np.min(x[:, 0]) - GRANULARITY
x0_max = np.max(x[:, 0]) + GRANULARITY
x1_min = np.min(x[:, 1]) - GRANULARITY
x1_max = np.max(x[:, 1]) + GRANULARITY
x0_range = np.arange(x0_min, x0_max, GRANULARITY)
x1_range = np.arange(x1_min, x1_max, GRANULARITY)

X_pairs = np.zeros((len(x0_range) * len(x1_range), 2))
i = 0
for i1 in range(len(x1_range)):
    for i0 in range(len(x0_range)):
        X_pairs[i] = np.array([x0_range[i0], x1_range[i1]])
        i += 1
y_hat_pairs = clf.predict(X_pairs)
print(X_pairs)
#print("mesh score = ", clf.score(X_pairs, y_hat_pairs))

x0_mesh, x1_mesh = np.meshgrid(x0_range, x1_range)
# print(y_hat_pairs)
y_hat_mesh = y_hat_pairs.reshape(x0_mesh.shape)

markers = ['o', '<', 's']
colours = [(1, 0, 0, 1), (0, 1, 0, 0.7), (0, 0, 1, 0.5)]
for i in range(len(x_train)):
    plt.plot(x_train[i, 0], x_train[i, 1], marker=markers[y_train[i]],
             markeredgecolor='w', markerfacecolor=colours[y_train[i]], markersize=9)
for i in range(len(x_test)):
    plt.plot(x_test[i, 0], x_test[i, 1], marker=markers[y_test[i]],
             markeredgecolor='w', markerfacecolor=colours[y_test[i]], markersize=9)
plt.pcolormesh(x0_mesh, x1_mesh, y_hat_mesh, shading="auto")
plt.set_cmap("Blues")
plt.show()

conf_scores = clf.decision_function(X_pairs)
y_binary = preprocess.label_binarize(y_hat_pairs, classes=sorted(set(y)))
false_pos = dict()
true_pos = dict()
for c in range(len(iris.target_names)):
    (false_pos[c], true_pos[c], tmp) = metrics.roc_curve(y_binary[:, c], conf_scores[:, c])
for c in range(len(iris.target_names)):
    plt.plot(false_pos[c], true_pos[c], label=iris.target_names[c])
plt.xlabel("false positive (FP) rate")
plt.ylabel("true positive (TP) rate")
#print(false_pos)
#print(true_pos)
plt.legend(loc="best")
plt.show()
