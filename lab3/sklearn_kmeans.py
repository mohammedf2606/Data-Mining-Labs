import numpy as np
import pandas as pd
import sklearn.cluster as cluster
import sklearn.datasets as data
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from prettytable import PrettyTable

from clustering import between_cluster_score, within_cluster_score

X, clusters = data._samples_generator.make_blobs(n_samples=100, n_features=2, cluster_std=1.0)

x = [i[0] for i in X]
y = [i[1] for i in X]
data = pd.DataFrame(zip(x, y))

within_list = []
between_list = []
SC_list = []
CH_list = []
table = PrettyTable()
table.field_names = ["K", "WC", "BC", "score", "inertia", "silhouette", "Calinski-Harabasz"]
K_range = list(range(2, 10))
for K in K_range:
    km = cluster.KMeans(n_clusters=K)
    km.fit(X)
    centres = km.cluster_centers_
    within = within_cluster_score(data, K, centres)
    within_list.append(within)
    between = between_cluster_score(K, centres)
    between_list.append(between)
    score = between/within
    SC = metrics.silhouette_score(X, km.labels_, metric="euclidean")
    SC_list.append(SC)
    CH = metrics.calinski_harabasz_score(X, km.labels_)
    CH_list.append(CH)
    table.add_row([K, within, between, score, km.inertia_, SC, CH])

plt.plot(K_range, within_list, label="Within Cluster")
plt.plot(K_range, between_list, label="Between Cluster")
plt.plot(K_range, SC_list, label="Silhouette score")
plt.plot(K_range, CH_list, label="Calinski-Harabasz score")
plt.legend(loc='best')
plt.show()

print(table)
