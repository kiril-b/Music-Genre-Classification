import umap
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import plotly.express as px
import plotly.offline as pyo
import matplotlib.pyplot as plt


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def reduce_dataset(df, features, label_column, genres, reducer, n_components, column_prefix):
    """
    Reduces the dimensionality of a dataset using a specified reducer technique and returns a DataFrame.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - features (list or None): List of feature columns to use for dimensionality reduction, or None to use all columns except label_column.
    - label_column (str): The column containing the labels for data points.
    - genres (list or None): List of genres to filter data points, or None to consider all data points.
    - reducer: The dimensionality reduction technique.
    - n_components (int): Number of components after dimensionality reduction.
    - column_prefix (str): Prefix for the column names of the reduced features.

    Returns:
    - pd.DataFrame: A DataFrame containing the reduced features and label_column.
    """
    if features is None:
        feature_df = df.drop(label_column, axis=1, inplace=False).copy()
    else:
        feature_df = df[features].copy()

    if genres is not None:
        genre_mask = df['genre'].isin(genres)
        feature_df = feature_df.loc[genre_mask, :]
        y = df.loc[genre_mask, 'genre']
    else:
        y = df[label_column]

    feature_df = StandardScaler().fit_transform(feature_df)
    reduced_df = reducer.fit_transform(feature_df)

    reduced_df_index = df[genre_mask].index if genres is not None else df.index
    reduced_df = pd.DataFrame(reduced_df, columns=[f'{column_prefix}{i + 1}' for i in range(n_components)], index=reduced_df_index)
    reduced_df[label_column] = y

    return reduced_df

def genre_2d_pca_visualization(df, features, genres, n_components=2, title=None):
    """
    Performs PCA-based dimensionality reduction on genre-labeled data and visualizes it using scatter and KDE plots.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - features (list): List of feature columns to use for PCA.
    - genres (list): List of genres to consider.
    - n_components (int): Number of components after PCA.
    - title (str or None): Title for the visualization.

    Returns:
    - None
    """
    reducer = PCA(n_components=n_components)
    reduced_df = reduce_dataset(df, features, 'genre', genres, reducer, n_components, column_prefix='PC_')

    fig, axs = plt.subplots(1, 2, figsize=(18, 7))
    sns.scatterplot(data=reduced_df, x='PC_1', y='PC_2', hue='genre', ax=axs[0])
    sns.kdeplot(data=reduced_df, x='PC_1', y='PC_2', hue='genre', ax=axs[1])
    if title is not None:
        fig.suptitle(title)


def genre_2d_umap_visualization(df, features, genres, n_components=2, n_neighbors=15, min_dist=0.1, metric='cosine'):
    """
    Performs UMAP-based dimensionality reduction on genre-labeled data and visualizes it using scatter and KDE plots.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - features (list): List of feature columns to use for UMAP.
    - genres (list): List of genres to consider.
    - n_components (int): Number of components after UMAP.
    - n_neighbors (int): Number of neighbors for UMAP.
    - min_dist (float): Minimum distance for UMAP.
    - metric (str): Distance metric for UMAP.

    Returns:
    - None
    """
    reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=n_components, metric=metric, min_dist=min_dist)
    reduced_df = reduce_dataset(df, features, 'genre', genres, reducer, n_components, column_prefix='x')

    _, axs = plt.subplots(1, 2, figsize=(18, 7))
    sns.scatterplot(data=reduced_df, x='x1', y='x2', hue='genre', ax=axs[0])
    sns.kdeplot(data=reduced_df, x='x1', y='x2', hue='genre', ax=axs[1])


def genre_3d_umap_visualization(df, label_column, features=None, genres=None, n_components=2, n_neighbors=15, min_dist=0.1, metric='cosine', output_file=None):
    """
    Performs 3D UMAP-based dimensionality reduction on genre-labeled data and visualizes it using a 3D scatter plot.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - label_column (str): The column containing the labels for data points.
    - features (list or None): List of feature columns to use for UMAP, or None to use all columns except label_column.
    - genres (list or None): List of genres to filter data points, or None to consider all data points (If genres is not None, label column should be 'genre').
    - n_components (int): Number of components after UMAP.
    - n_neighbors (int): Number of neighbors for UMAP.
    - min_dist (float): Minimum distance for UMAP.
    - metric (str): Distance metric for UMAP.
    - output_file (str or None): File name to save the plot, or None to only display the plot.

    Returns:
    - None
    """
     
    if genres is not None and label_column != 'genre':
        raise Exception("If 'genres' is not None, label column should be 'genre'")

    reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=n_components, metric=metric, min_dist=min_dist)
    reduced_df = reduce_dataset(df, features, label_column, genres, reducer, n_components, column_prefix='x')
    
    fig_3d = px.scatter_3d(
        reduced_df.drop(label_column, axis=1, inplace=False), 
        x='x1', y='x2', z='x3',
        color=reduced_df[label_column], 
    )
    fig_3d.update_traces(marker_size=4)

    pyo.init_notebook_mode(connected=True)
    pyo.iplot(fig_3d)

    if output_file:
        pyo.plot(fig_3d, filename=f'./plots/{output_file}')


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


def plot_outliers_per_genre(pca_df, outliers_indexes):
    """
    Plots outliers detected within different genres using a scatter plot.

    Parameters:
    pca_df (DataFrame): A DataFrame containing PCA-transformed data along with genre labels.
    outliers_indexes (list or array-like): Indexes of the outliers to be highlighted in the plot.

    Returns:
    None
    """
    fig, axs = plt.subplots(4, 4, figsize=(20, 20))
    axs_tuples = [(i,j) for i in range (0, 4) for j in range (0, 4)]
    fig.suptitle('Detected outliers within genres')
    k = 0

    for genre in pca_df['genre'].unique():
        df_genre = pca_df.query("genre == @genre")
        i, j = axs_tuples[k]
        ax = axs[i][j]

        sns.scatterplot(data=df_genre, x='PC1', y='PC2', ax=ax)
        ax.scatter(df_genre.loc[outliers_indexes,'PC1'], df_genre.loc[outliers_indexes,'PC2'], color='purple', marker='+', s=30)

        ax.set_title(f'{genre}')

        k += 1


def plot_num_outliers_per_genre(df, outliers_indexes):
    """
    Plots the number of outliers detected per genre using a bar plot.

    Parameters:
    df (DataFrame): A DataFrame containing data with genre labels.
    outliers_indexes (list or array-like): Indexes of the outliers to be counted and plotted.

    Returns:
    None
    """
    outliers_by_genre = df.loc[outliers_indexes]['genre'].value_counts()
    plt.figure(figsize=(10, 9))
    sns.barplot(y=outliers_by_genre.index, x=outliers_by_genre.values, color='#00B3B3')
    plt.title('Number of outliers per genre')
    plt.show()


def plot_cluster_similarities(label, label_value, similarities_df, ax):
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