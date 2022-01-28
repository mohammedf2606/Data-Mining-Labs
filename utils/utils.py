import random as r


def dot_product(vec1, vec2):
    assert len(vec1) == len(vec2)
    running_total = 0
    for i in range(len(vec1)):
        running_total += vec1[i] * vec2[i]
    return running_total


def compute_r2(predicted, actual):
    sum_squared_error = 0
    population_variance = 0
    y_mean = sum(actual) / len(actual)
    for j in range(len(actual)):
        sum_squared_error += (actual[j] - predicted[j]) ** 2
        population_variance += (actual[j] - y_mean) ** 2
    R_2 = 1 - (sum_squared_error / population_variance)
    return R_2


# returns lists ([trainset], [testset]) of tuples [(x1,y1) ... (xn, yn)]
def random_partition(x_rand, y_rand):
    train_size = int(len(x_rand) * 0.9)
    all_indices = list(range(len(x_rand)))
    random_train_indices = r.sample(all_indices, train_size)
    random_test_indices = list(set(all_indices) - set(random_train_indices))
    index_values = list(enumerate(zip(x_rand, y_rand)))  # Collection(index, (x,y))
    trainset = [index_values[i][1] for i in random_train_indices]  # (x,y)
    testset = [index_values[i][1] for i in random_test_indices]

    return trainset, testset
