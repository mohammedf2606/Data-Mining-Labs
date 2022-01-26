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
