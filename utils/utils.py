import pandas as pd
import matplotlib.pyplot as plt
import ast
import seaborn as sns
import scipy.stats as stats


class MissingValuesUtil:
    def __init__(self, df):
        self.df = df
        self.mv_df = pd.DataFrame([df.isna().sum(), df.isna().mean() * 100.0],
                                  index=['count', 'proportion (%)'],
                                  columns=df.columns).T
        self.rows_with_mv = self.mv_df['count'] != 0

    def get_mvdf(self):
        return self.mv_df

    def get_mvdf_missing_only(self):
        return self.mv_df[self.rows_with_mv]

    def features_mv_only(self):
        return self.mv_df[self.rows_with_mv].index

    def features_no_mv(self):
        return self.mv_df[~self.rows_with_mv].index

    def barplot_mv(self, w, h):
        plt.figure(figsize=(w, h))
        temp_df = self.get_mvdf_missing_only().sort_values(by='proportion (%)', ascending=False)
        sns.barplot(y=temp_df.index, x=temp_df['proportion (%)'], color='#00B3B3')
        plt.title('Proportion of missing values for features that have missing values')
        plt.show()


def corr_heatmap(df, axis_label, title, method='pearson', figsize=(25, 18)):
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
    columns = df.shape[1]
    _, ax = plt.subplots(1, columns, figsize=figsize)
    ax_idx = 0
    for col in df.columns:
        stats.probplot(df[col], dist='norm', plot=ax[ax_idx])
        ax[ax_idx].set_title('Q-Q plot for ' + col)
        ax_idx += 1
    plt.show()
