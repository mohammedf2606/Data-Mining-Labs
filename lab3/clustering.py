import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import sklearn.datasets as data


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
    square_dist = distances ** 2
    score = sum(sum(square_dist))
    return score


X, clusters = data._samples_generator.make_blobs(n_samples=100, n_features=2, cluster_std=1.0)

x = [i[0] for i in X]
y = [i[1] for i in X]
data = pd.DataFrame(zip(x, y))
K = 4
labels, centres = kmeans(data, K, 100)

print("Within Cluster score: " + str(within_cluster_score(data, K, centres)))

u_labels = np.unique(labels)
print(data)
for i in u_labels:
    plt.scatter(data.loc[labels == i, 0], data.loc[labels == i, 1], label=i)
plt.legend(loc='best')
plt.show()
