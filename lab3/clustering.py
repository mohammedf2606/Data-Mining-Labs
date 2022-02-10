from pprint import pp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import sklearn.datasets as datasets


def between_cluster_score(K, centres):
    distances = cdist(centres, centres, 'euclidean')
    square_dist = [(i ** 2).tolist() for i in distances]
    sum_list = [sum(d) for d in square_dist]
    return sum(sum_list)/2


def kmeans(x, K, no_of_iter):
    centroids = x.sample(n=K)
    distances = cdist(x, centroids, 'euclidean')

    points = np.array([np.argmin(i) for i in distances])

    for _ in range(no_of_iter):
        centroids = []
        for index in range(K):
            temp_centre = x[points == index].mean(axis=0)
            centroids.append(temp_centre)
        centroids = np.vstack(centroids)

        distances = cdist(x, centroids, 'euclidean')
        points = np.array([np.argmin(i) for i in distances])

    return points, centroids


def within_cluster_score(x, K, centres):
    distances = cdist(x, centres, 'euclidean')
    square_dist = [i ** 2 for i in distances]
    sum_cluster = [min(i) for i in square_dist]
    return sum(sum_cluster)


def main():
    X, clusters = datasets._samples_generator.make_blobs(n_samples=100, n_features=2, cluster_std=1.0)

    x = [i[0] for i in X]
    y = [i[1] for i in X]
    data = pd.DataFrame(zip(x, y))

    within_list = []
    between_list = []
    K_range = list(range(2, 9))
    for K in K_range:
        labels, centres = kmeans(data, K, 1000)

        u_labels = np.unique(labels)
        # for i in u_labels:
        #     plt.scatter(data.loc[labels == i, 0], data.loc[labels == i, 1], label=i)
        # plt.legend(loc='best')
        # plt.show()

        within = within_cluster_score(data, K, centres)
        within_list.append(within)
        print("Within Cluster score: " + str(within))
        between = between_cluster_score(K, centres)
        between_list.append(between)
        print("Between Cluster score: " + str(between))
        # print("Score: " + str(between/within))

    plt.plot(K_range, within_list, label="Within Cluster")
    plt.plot(K_range, between_list, label="Between Cluster")
    plt.legend(loc='best')
    plt.show()


if __name__ == "__main__":
    main()

