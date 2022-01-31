from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def calculate_most_frequent_class(X, target, feature, val):
    class_counter = dict()
    for sample, y in zip(x, target):
        if sample[feature] == val:
            class_counter[y] += 1

def train(x_train, y_train, feature):
    n_samples, n_features = x_train.shape
    values = set(x[:, feature])
    predictors = dict()
    errors = []
    for val in values:
        most_frequent_class, error = calculate_most_frequent_class(x_train, y_train, feature, val)

iris = load_iris()
print("classes = ", iris.target_names)
print("attributes = ", iris.feature_names)
M = len(iris.data)
print("number of instances = %d" % M)

sepal_length = [x[0] for x in iris.data]
sepal_width = [x[1] for x in iris.data]

target_classes = iris.targets

x_train, x_test, y_train, y_test = train_test_split(sepal_length, target_classes, random_state = 14)

print(sepal_length)

x_y_pairs = []
c_freq = dict()

predictors = {feature: train(x_train, y_train, feature) for feature in range(x_train.shape[1])}
errors = {variable: error for variable, (mapping, error) in predictors}

for x in sepal_length:
    for y in target_classes:
        x_y_pairs.append((x, y))
    c_freq[x] =

