from typing import Union
import umap
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import plotly.express as px
import plotly.offline as pyo
import matplotlib.pyplot as plt


from matplotlib.axes import Axes
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def reduce_dataset(df: pd.DataFrame, 
                   label_column: str, 
                   reducer: any, 
                   n_components: int, 
                   column_prefix: str, 
                   genres: list[str] = None, 
                   features: list[str] = None) -> pd.DataFrame:
    """
    Reduce dataset dimensions using a dimensionality reduction technique.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        label_column (str): The column containing the labels.
        reducer (any): An instance of a dimensionality reduction technique.
        n_components (int): Number of components for reduction.
        column_prefix (str): Prefix for the reduced columns.
        genres (list[str], optional): List of genres to consider. Defaults to None.
        features (list[str], optional): List of features to consider. Defaults to None.
    
    Returns:
        pd.DataFrame: Reduced DataFrame with specified columns.
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

def genre_2d_pca_visualization(df: pd.DataFrame, features: list[str], genres: list[str], n_components: int = 2, title: str = None) -> None:
    """
    Visualize genres in 2D using PCA dimensionality reduction.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        features (list[str]): List of features for analysis.
        genres (list[str]): List of genres to visualize.
        n_components (int, optional): Number of components for PCA. Defaults to 2.
        title (str, optional): Title for the plot. Defaults to None.
    """
    reducer = PCA(n_components=n_components)
    reduced_df = reduce_dataset(df, 'genre', reducer, n_components, column_prefix='PC_', genres=genres, features=features)

    fig, axs = plt.subplots(1, 2, figsize=(18, 7))
    sns.scatterplot(data=reduced_df, x='PC_1', y='PC_2', hue='genre', ax=axs[0])
    sns.kdeplot(data=reduced_df, x='PC_1', y='PC_2', hue='genre', ax=axs[1])
    if title is not None:
        fig.suptitle(title)


def genre_2d_umap_visualization(df: pd.DataFrame, features: list[str], 
                                genres: list[str], n_components:int = 2, 
                                n_neighbors: int = 15, min_dist: float = 0.1, metric: str = 'cosine') -> None:
    """
    Visualize genres in 2D using UMAP dimensionality reduction.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        features (list[str]): List of features for analysis.
        genres (list[str]): List of genres to visualize.
        n_components (int, optional): Number of components for UMAP. Defaults to 2.
        n_neighbors (int, optional): Number of neighbors for UMAP. Defaults to 15.
        min_dist (float, optional): Minimum distance for UMAP. Defaults to 0.1.
        metric (str, optional): Metric for UMAP distance calculation. Defaults to 'cosine'.
    """
    reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=n_components, metric=metric, min_dist=min_dist)
    reduced_df = reduce_dataset(df, 'genre', reducer, n_components, column_prefix='x', genres=genres, features=features)

    _, axs = plt.subplots(1, 2, figsize=(18, 7))
    sns.scatterplot(data=reduced_df, x='x1', y='x2', hue='genre', ax=axs[0])
    sns.kdeplot(data=reduced_df, x='x1', y='x2', hue='genre', ax=axs[1])


def genre_3d_umap_visualization(df: pd.DataFrame, label_column: str, 
                                n_components:int = 2, n_neighbors:int = 15, min_dist: float = 0.1, metric: str = 'cosine',
                                  features: list[str] = None, genres: list[str] = None, output_file: str = None) -> None:
    """
    Visualize genres in 3D using UMAP dimensionality reduction.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        label_column (str): The column containing the labels.
        n_components (int, optional): Number of components for UMAP. Defaults to 2.
        n_neighbors (int, optional): Number of neighbors for UMAP. Defaults to 15.
        min_dist (float, optional): Minimum distance for UMAP. Defaults to 0.1.
        metric (str, optional): Metric for UMAP distance calculation. Defaults to 'cosine'.
        features (list[str], optional): List of features to consider. Defaults to None.
        genres (list[str], optional): List of genres to consider. Defaults to None.
        output_file (str, optional): Output file name for saving the plot. Defaults to None.
    """
    if genres is not None and label_column != 'genre':
        raise Exception("If 'genres' is not None, label column should be 'genre'")

    reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=n_components, metric=metric, min_dist=min_dist)
    reduced_df = reduce_dataset(df, label_column, reducer, n_components, column_prefix='x', genres=genres, features=features)
    
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


def corr_heatmap(df: pd.DataFrame, axis_label: str, title: str, method: str = 'pearson', figsize: tuple[int, int] = (25, 18)) -> None:
    """
    Plot a correlation heatmap for the DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        axis_label (str): Label for x and y axes.
        title (str): Title of the plot.
        method (str, optional): Correlation method. Defaults to 'pearson'.
        figsize (tuple[int, int], optional): Figure size. Defaults to (25, 18).
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


def plot_qq(df: pd.DataFrame, figsize: tuple[int, int]) -> None:
    """
    Plot Q-Q plots for each column in the DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        figsize (tuple[int, int]): Figure size.
    """
    columns = df.shape[1]
    _, ax = plt.subplots(1, columns, figsize=figsize)
    ax_idx = 0
    for col in df.columns:
        stats.probplot(df[col], dist='norm', plot=ax[ax_idx])
        ax[ax_idx].set_title('Q-Q plot for ' + col)
        ax_idx += 1
    plt.show()


def plot_outliers_per_genre(pca_df: pd.DataFrame, outliers_indexes: Union[pd.Index, list]) -> None:
    """
    Plot scatter plots of PCA components with detected outliers highlighted.
    
    Args:
        pca_df (pd.DataFrame): DataFrame containing PCA components.
        outliers_indexes (Union[pd.Index, list]): Indexes of detected outliers.
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


def plot_num_outliers_per_genre(df: pd.DataFrame, outliers_indexes: Union[pd.Index, list]) -> None:
    """
    Plot a bar plot of the number of outliers per genre.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        outliers_indexes (Union[pd.Index, list]): Indexes of detected outliers.
    """
    outliers_by_genre = df.loc[outliers_indexes]['genre'].value_counts()
    plt.figure(figsize=(10, 9))
    sns.barplot(y=outliers_by_genre.index, x=outliers_by_genre.values, color='#00B3B3')
    plt.title('Number of outliers per genre')
    plt.show()


def plot_cluster_similarities(label: str, label_value: str, similarities_df: pd.DataFrame, ax: Axes) -> None:
    """
    Plot bar plot of cluster similarities for a specific label value.
    
    Args:
        label (str): Label for the clusters (e.g., 'genre').
        label_value (str): Value of the label to plot similarities for.
        similarities_df (pd.DataFrame): DataFrame containing cluster similarity data.
        ax (Axes): Matplotlib axis to plot on.
    """
    similarities_df = similarities_df.query(f'{label}_1 == "{label_value}" or {label}_2 == "{label_value}"').sort_values(by='similarity', ascending=False)
    similarities_df[label] = similarities_df[f'{label}_1'].str.cat(similarities_df[f'{label}_2'], sep=' <-> ')
    ax.set_title(f'Genre similarity for {label_value}')
    sns.barplot(data=similarities_df, x='similarity', y=f'{label}', ax=ax, color='#00B3B3')