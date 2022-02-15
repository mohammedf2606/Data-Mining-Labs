# Part 2: Cluster Analysis

# Return a pandas dataframe containing the data set that needs to be extracted from the data_file.
# data_file will be populated with the string 'wholesale_customers.csv'.
import pandas as pd
from sklearn import metrics
from sklearn.cluster import KMeans, AgglomerativeClustering


def read_csv_2(data_file):
    df = pd.read_csv(data_file)
    return df.drop(columns=["Channel", "Region"])


# Return a pandas dataframe with summary statistics of the data.
# Namely, 'mean', 'std' (standard deviation), 'min', and 'max' for each attribute.
# These strings index the new dataframe columns. 
# Each row should correspond to an attribute in the original data and be indexed with the attribute name.
def summary_statistics(df):
    mean = round(df.mean().astype(int))
    std = round(df.std().astype(int))
    min = df.min()
    max = df.max()
    return pd.DataFrame([mean, std, min, max]).rename(index={0: "mean", 1: "std", 2: "min", 3: "max"})


# Given a dataframe df with numeric values, return a dataframe (new copy)
# where each attribute value is subtracted by the mean and then divided by the
# standard deviation for that attribute.
def standardize(df):
    summary = summary_statistics(df)
    new_df = (df - summary.loc["mean"])/summary.loc["std"]
    return new_df


# Given a dataframe df and a number of clusters k, return a pandas series y
# specifying an assignment of instances to clusters, using kmeans.
# y should contain values in the set {0,1,...,k-1}.
def kmeans(df, k):
    km = KMeans(n_clusters=k)
    km.fit(df)
    return km.predict(df)


# Given a dataframe df and a number of clusters k, return a pandas series y
# specifying an assignment of instances to clusters, using agglomerative hierarchical clustering.
# y should contain values from the set {0,1,...,k-1}.
def agglomerative(df, k):
    ac = AgglomerativeClustering(n_clusters=k, linkage='average', affinity='euclidean')
    y_pred = ac.fit_predict(df)
    return y_pred


# Given a data set X and an assignment to clusters y
# return the Solhouette score of the clustering.
def clustering_score(X, y):
    return metrics.silhouette_score(X, y, metric="euclidean")


# Perform the cluster evaluation described in the coursework description.
# Given the dataframe df with the data to be clustered,
# return a pandas dataframe with an entry for each clustering algorithm execution.
# Each entry should contain the: 
# 'Algorithm' name: either 'Kmeans' or 'Agglomerative', 
# 'data' type: either 'Original' or 'Standardized',
# 'k': the number of clusters produced,
# 'Silhouette Score': for evaluating the resulting set of clusters.
def cluster_evaluation(df):
    columns = ["Algorithm", "data", "k", "Silhouette Score"]
    k_list = [3, 5, 10]
    eval_df = pd.DataFrame(columns=columns)
    i = 0
    for k in k_list:
        standardized = standardize(df)

        k_means = kmeans(df, k)
        k_means_stand = kmeans(standardized, k)
        sc_standard_kmeans = clustering_score(standardized, k_means_stand)
        sc_original_kmeans = clustering_score(df, k_means)

        agglo = agglomerative(df, k)
        agglo_stand = agglomerative(standardized, k)
        sc_standard_agglo = clustering_score(standardized, agglo_stand)
        sc_original_agglo = clustering_score(df, agglo)

        eval_df.loc[i] = ["Kmeans", "Original", k, sc_original_kmeans]
        eval_df.loc[i+1] = ["Kmeans", "Standardized", k, sc_standard_kmeans]
        eval_df.loc[i+2] = ["Agglomerative", "Original", k, sc_original_agglo]
        eval_df.loc[i+3] = ["Agglomerative", "Standardized", k, sc_standard_agglo]
        i += 4

    return eval_df


# Given the performance evaluation dataframe produced by the cluster_evaluation function,
# return the best computed Silhouette score.
def best_clustering_score(rdf):
    return rdf.max()


# Run some clustering algorithm of your choice with k=3 and generate a scatter plot for each pair of attributes.
# Data points in different clusters should appear with different colors.
def scatter_plots(df):
    pass


if __name__ == "__main__":
    df = read_csv_2('./data/wholesale_customers.csv')
    print(summary_statistics(df))
    print(cluster_evaluation(df))


