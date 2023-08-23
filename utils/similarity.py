import os

import numpy as np
import pandas as pd
import seaborn as sns

from itertools import combinations

from sklearn.metrics.pairwise import euclidean_distances


def rbf_kernel(x, y, gamma):
    """
    Compute the Radial Basis Function (RBF) kernel between two sets of samples.

    Args:
        x (array-like): The first set of samples.
        y (array-like): The second set of samples.
        gamma (float): The kernel parameter.

    Returns:
        ndarray: The RBF kernel matrix.
    """
    return np.exp(-gamma * euclidean_distances(x, y)**2)

def average_similarity(cluster1, cluster2):
    """
    Calculate the average similarity between two clusters using the RBF kernel.

    Args:
        cluster1 (pd.DataFrame): The first cluster.
        cluster2 (pd.DataFrame): The second cluster.

    Returns:
        float: The average similarity between the clusters.
    """
    similarity_matrix = rbf_kernel(cluster1, cluster2, 1 / (len(cluster1.columns)*4))
    total_similarity = np.sum(similarity_matrix)
    average_similarity = total_similarity / (len(cluster1) * len(cluster2))
    return average_similarity


def calculate_cluster_similarities(df, label, features):
    """
    Calculate pairwise cluster similarities based on average similarity.

    Args:
        df (pd.DataFrame): The input DataFrame.
        label (str): The column name for cluster labels.
        features (list): The list of feature column names.

    Returns:
        pd.DataFrame: A DataFrame containing pairwise cluster similarities.
    """
    results = list()
    for c1, c2 in list(combinations(df[label].unique(), 2)):
        c1_features = df[df[label] == c1][features]
        c2_features = df[df[label] == c2][features]
        results.append([c1, c2, average_similarity(c1_features, c2_features)])

    return pd.DataFrame(results, columns=[f'{label}_1', f'{label}_2', 'similarity'])



