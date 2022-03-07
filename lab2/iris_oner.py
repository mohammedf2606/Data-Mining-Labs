import random
from collections import Counter
from prettytable import PrettyTable
from sklearn.datasets import load_iris


def make_table(feature_vectors, targets):
    class_counter = Counter(zip(feature_vectors, targets))
    class_counter = dict(sorted(class_counter.items()))
    class_occurrences = dict()
    class_poss = [0, 1, 2]
    for x, count in class_counter.items():
        if x[0] not in class_occurrences:
            class_occurrences[x[0]] = [0 for i in range(len(class_poss))]
        temp_list = class_occurrences[x[0]]
        temp_list[x[1]] = temp_list[x[1]] + count
        class_occurrences[x[0]] = temp_list
    rows = [(i, j, j.index(max(j))) for (i, j) in list(class_occurrences.items())]
    table = PrettyTable()
    table.field_names = ["Sepal length (cm)", "class frequencies: { 0, 1, 2 }", "most frequent"]
    for el in rows:
        table.add_row(el)
    return table, rows


def make_matrix(length, width):
    classification_matrix = {}
    for i in length:
        for j in width:
            if i[2] != j[2]:
                if i[1][i[2]] > j[1][j[2]]:
                    classification_matrix[(i[0], j[0])] = i[2]
                elif i[1][i[2]] < j[1][j[2]]:
                    classification_matrix[(i[0], j[0])] = j[2]
                else:
                    if random.random() < 0.5:
                        classification_matrix[(i[0], j[0])] = i[2]
                    else:
                        classification_matrix[(i[0], j[0])] = j[2]
            else:
                classification_matrix[(i[0], j[0])] = i[2]
    return classification_matrix


iris = load_iris()
print("classes = ", iris.target_names)
print("attributes = ", iris.feature_names)
M = len(iris.data)
print("number of instances = %d" % M)

sepal_length = [x[0] for x in iris.data]
sepal_width = [x[1] for x in iris.data]

target_classes = list(iris.target)

table_length, length_data = make_table(sepal_length, target_classes)
table_width, width_data = make_table(sepal_width, target_classes)

print(table_length)
print(table_width)

# (length, width): most_frequent
most_frequent_pairs = make_matrix(length_data, width_data)

print(most_frequent_pairs)

correct_predictions = [i == j for i, j in zip(most_frequent_pairs.values(), target_classes)]
score = sum(correct_predictions) / len(correct_predictions)
error = (len(correct_predictions) - sum(correct_predictions)) / len(correct_predictions)
print("Score: " + str(score) + "\n")
print("Resubstitution error: " + str(error) + "\n")

