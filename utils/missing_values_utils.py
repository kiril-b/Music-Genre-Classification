import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class MissingValuesUtil:
    """A utility class for analyzing missing values in a pandas DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing data with missing values.

    Attributes:
        df (pd.DataFrame): The input DataFrame.
        mv_df (pd.DataFrame): A DataFrame containing information about missing values.
                            It includes the count and proportion of missing values for each feature.
        rows_with_mv (pd.Series): A boolean series indicating whether a feature has missing values.

    Methods:
        get_mvdf(): Retrieve the DataFrame containing information about missing values.
        get_mvdf_missing_only(): Retrieve a DataFrame with information about missing values,
                                limited to features with missing values.
        features_mv_only(): Retrieve an index of rows with missing values.
        features_no_mv(): Retrieve an index of rows without missing values.
        barplot_mv(w, h): Generate and display a bar plot showing the proportion of missing values
                        for features with missing values.

    """
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.mv_df = pd.DataFrame(
                [df.isna().sum(), df.isna().mean() * 100.0],
                index=['count', 'proportion (%)'],
                columns=df.columns
            ).T
        self.rows_with_mv = self.mv_df['count'] != 0

    def get_mvdf(self) -> pd.DataFrame:
        """
        Returns the DataFrame containing information about missing values.

        Returns:
        pd.DataFrame: A DataFrame with columns 'count' and 'proportion (%)'
                      indicating the count and proportion of missing values for each feature.
        """
        return self.mv_df

    def get_mvdf_missing_only(self) -> pd.DataFrame:
        """
        Returns a DataFrame with information about missing values, limited to features with missing values.

        Returns:
        pd.DataFrame: A DataFrame containing missing value information for features with missing values.
        """
        return self.mv_df[self.rows_with_mv]

    def features_mv_only(self) -> pd.Index:
        """
        Returns an index of rows with missing values.

        Returns:
        pd.Index: A pandas Index object
        """
        return self.mv_df[self.rows_with_mv].index

    def features_no_mv(self) -> pd.Index:
        """
        Returns an index of rows without missing values.

        Returns:
        pd.Index: A pandas Index object
        """
        return self.mv_df[~self.rows_with_mv].index

    def barplot_mv(self, w: int, h: int) -> None:
        """
        Generates and displays a bar plot showing the proportion of missing values for features with missing values.

        Parameters:
        w (int): Width of the plot.
        h (int): Height of the plot.

        Returns:
        None
        """
        plt.figure(figsize=(w, h))
        temp_df = self.get_mvdf_missing_only().sort_values(by='proportion (%)', ascending=False)
        sns.barplot(y=temp_df.index, x=temp_df['proportion (%)'], color='#00B3B3')
        plt.title('Proportion of missing values for features that have missing values')
        plt.show()

