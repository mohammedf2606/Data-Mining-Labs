import numpy as np
import pandas as pd
import sklearn.cluster as cluster
import sklearn.datasets as data
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

from clustering import between_cluster_score

X, clusters = data._samples_generator.make_blobs(n_samples=100, n_features=2, cluster_std=1.0)

x = [i[0] for i in X]
y = [i[1] for i in X]
data = pd.DataFrame(zip(x, y))

within_list = []
between_list = []
for K in range(2, 10):
    km = cluster.KMeans(n_clusters=K)
    km.fit(X)
    centres = km.cluster_centers_
    # labels = km.labels_
    # u_labels = np.unique(labels)
    # # for i in km.labels_:
    # #     plt.scatter(data.loc[labels == i, 0], data.loc[labels == i, 1], label=i)
    # # plt.legend(loc='best')
    # # plt.show()

    within_list.append(km.inertia_)
    # print("Within Cluster score: " + str(within))
    between = between_cluster_score(K, centres)
    between_list.append(between)
    # print("Between Cluster " + str(between))
    # print("Score: " + str(between/within))

plt.plot(within_list, label="Within Cluster")
plt.plot(between_list, label="Between Cluster")
plt.legend(loc='best')
plt.show()
