from itertools import combinations

import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt

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


def plot_similarities(label, label_value, similarities_df, ax):
    """
    Plot cluster similarities for a specific cluster label.

    Args:
        label (str): The cluster label column name.
        label_value (str): The specific cluster label value to plot.
        similarities_df (pd.DataFrame): DataFrame containing cluster similarities.
        ax (matplotlib.axes.Axes): The axis to plot the bar chart.
    """
    similarities_df = similarities_df.query(f'{label}_1 == "{label_value}" or {label}_2 == "{label_value}"').sort_values(by='similarity', ascending=False)
    similarities_df[label] = similarities_df[f'{label}_1'].str.cat(similarities_df[f'{label}_2'], sep=' <-> ')
    ax.set_title(f'Genre similarity for {label_value}')
    sns.barplot(data=similarities_df, x='similarity', y=f'{label}', ax=ax, color='#00B3B3')


def corr_heatmap(df, axis_label, title, method='pearson', figsize=(25, 18)):
    """
    Create a heatmap to visualize the correlation between features in a DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        axis_label (str): Label for the axis.
        title (str): Title of the heatmap.
        method (str): Correlation method (default is 'pearson').
        figsize (tuple): Figure size (width, height).
    """
    plt.figure(figsize=figsize)
    sns.heatmap(
        df.corr(method=method),
        cmap='RdBu',
        annot=True,
        vmin=-1,
        vmax=1
    )
    plt.xlabel(axis_label)
    plt.ylabel(axis_label)
    plt.title(title)
    plt.show()


def plot_qq(df, figsize):
    """
    Create Q-Q plots to visualize the distribution of features in comparison to a normal distribution.

    Args:
        df (pd.DataFrame): The input DataFrame.
        figsize (tuple): Figure size (width, height).
    """
    columns = df.shape[1]
    _, ax = plt.subplots(1, columns, figsize=figsize)
    ax_idx = 0
    for col in df.columns:
        stats.probplot(df[col], dist='norm', plot=ax[ax_idx])
        ax[ax_idx].set_title('Q-Q plot for ' + col)
        ax_idx += 1
    plt.show()
