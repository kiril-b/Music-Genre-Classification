import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class MissingValuesUtil:
    """
    A utility class for analyzing missing values in a DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Attributes:
        df (pd.DataFrame): The input DataFrame.
        mv_df (pd.DataFrame): DataFrame containing information about missing values.
        rows_with_mv (pd.Series): Boolean series indicating rows with missing values.

    Methods:
        get_mvdf(): Get the DataFrame containing missing value information.
        get_mvdf_missing_only(): Get a subset of the DataFrame containing only missing value information.
        features_mv_only(): Get a list of feature names with missing values.
        features_no_mv(): Get a list of feature names without missing values.
        barplot_mv(w, h): Create a bar plot showing the proportion of missing values for features with missing values.
    """
    def __init__(self, df):
        """
        Initialize the MissingValuesUtil instance.

        Args:
            df (pd.DataFrame): The input DataFrame.
        """
        self.df = df
        self.mv_df = pd.DataFrame([df.isna().sum(), df.isna().mean() * 100.0],
                                  index=['count', 'proportion (%)'],
                                  columns=df.columns).T
        self.rows_with_mv = self.mv_df['count'] != 0

    def get_mvdf(self):
        """
        Get the DataFrame containing missing value information.

        Returns:
            pd.DataFrame: DataFrame containing missing value information.
        """
        return self.mv_df

    def get_mvdf_missing_only(self):
        """
        Get a subset of the DataFrame containing only missing value information.

        Returns:
            pd.DataFrame: Subset of the DataFrame with missing value information.
        """
        return self.mv_df[self.rows_with_mv]

    def features_mv_only(self):
        """
        Get a list of feature names with missing values.

        Returns:
            pd.Index: Index containing feature names with missing values.
        """
        return self.mv_df[self.rows_with_mv].index

    def features_no_mv(self):
        """
        Get a list of feature names without missing values.

        Returns:
            pd.Index: Index containing feature names without missing values.
        """
        return self.mv_df[~self.rows_with_mv].index

    def barplot_mv(self, w, h):
        """
        Create a bar plot showing the proportion of missing values for features with missing values.

        Args:
            w (int): Width of the plot.
            h (int): Height of the plot.
        """
        plt.figure(figsize=(w, h))
        temp_df = self.get_mvdf_missing_only().sort_values(by='proportion (%)', ascending=False)
        sns.barplot(y=temp_df.index, x=temp_df['proportion (%)'], color='#00B3B3')
        plt.title('Proportion of missing values for features that have missing values')
        plt.show()

