import sklearn.cluster as cluster
import sklearn.datasets as data
import matplotlib.pyplot as plt
import numpy as np

X, clusters = data._samples_generator.make_blobs(n_samples=100, n_features=2, cluster_std=1.0)

K_range = list(range(2, 10))
for K in K_range:
    ac = cluster.AgglomerativeClustering(n_clusters=K, linkage='average', affinity='euclidean')
    ac.fit(X)
