import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


class ClassificationEvaluation():
    """
    A class for evaluating classification models and generating evaluation metrics.

    Args:
        model_name (str): The name of the classification model.
        model: The trained classification model.
        train_X (pd.DataFrame): The training data features.
        train_y (pd.Series): The training data labels.
        val_X (pd.DataFrame): The validation data features.
        val_y (pd.Series): The validation data labels.

    Attributes:
        model_name (str): The name of the classification model.
        model: The trained classification model.
        train_X (pd.DataFrame): The training data features.
        train_y (pd.Series): The training data labels.
        val_X (pd.DataFrame): The validation data features.
        val_y (pd.Series): The validation data labels.
        train_y_pred (np.ndarray): Predicted labels on the training data.
        val_y_pred (np.ndarray): Predicted labels on the validation data.

    Methods:
        get_scores(on_sets, average_type): Get various classification metrics.
        plot_confusion_matrix(ax, on_set='validation'): Plot a normalized confusion matrix.
        plot_metrics_per_class(ax, on_set='validation', horizontal=True): Plot precision, recall, and F1-score per class.
    """
    def __init__(self, model_name, model, train_X, train_y, val_X, val_y):
        """
        Initialize the ClassificationEvaluation instance.

        Args:
            model_name (str): The name of the classification model.
            model: The trained classification model.
            train_X (pd.DataFrame): The training data features.
            train_y (pd.Series): The training data labels.
            val_X (pd.DataFrame): The validation data features.
            val_y (pd.Series): The validation data labels.
        """
        self.model_name = model_name
        self.model = model
        self.train_X = train_X
        self.train_y = train_y
        self.val_X = val_X
        self.val_y = val_y
        self.train_y_pred = self.model.predict(self.train_X)
        self.val_y_pred = self.model.predict(self.val_X)

    def get_scores(self, on_sets, average_type='weighted'):
        """
        Calculate classification metrics for specified datasets.

        Args:
            on_sets (list): List of datasets to evaluate (e.g., ['train', 'validation']).
            average_type (str): Type of averaging for precision, recall, and F1-score (default is 'weighted').

        Returns:
            pd.DataFrame: DataFrame containing classification metrics for each dataset.
        """
        if on_sets is None:
            raise Exception("You need to pass an array on_sets, containing strings 'train' and/or 'validation'")

        data = []
        index = ['accuracy', f'precision ({average_type})', f'recall ({average_type})', f'f1 ({average_type})']

        for dataset in on_sets:
            if dataset == 'train':
                data.append([accuracy_score(self.train_y, self.train_y_pred),
                             precision_score(self.train_y, self.train_y_pred, average=average_type, zero_division=0),
                             recall_score(self.train_y, self.train_y_pred, average=average_type),
                             f1_score(self.train_y, self.train_y_pred, average=average_type)])
            elif dataset == 'validation':
                data.append([accuracy_score(self.val_y, self.val_y_pred),
                             precision_score(self.val_y, self.val_y_pred, average=average_type, zero_division=0),
                             recall_score(self.val_y, self.val_y_pred, average=average_type),
                             f1_score(self.val_y, self.val_y_pred, average=average_type)])
            else:
                raise Exception("You need to pass an array on_sets, containing strings 'train' and/or 'validation'")

        return pd.DataFrame(data, columns=index, index=on_sets).T

    def plot_confusion_matrix(self, ax, on_set='validation'):
        """
        Plot a normalized confusion matrix.

        Args:
            ax: Matplotlib axis to draw the plot on.
            on_set (str): Dataset to evaluate (either 'train' or 'validation').
        """
        if on_set not in ['train', 'validation']:
            raise Exception("The parameter on_set can be either 'train' or 'validation'")

        y = self.train_y if on_set == 'train' else self.val_y
        y_pred = self.train_y_pred if on_set == 'train' else self.val_y_pred

        cm = confusion_matrix(y, y_pred, normalize='true')
        sns.heatmap(
            cm,
            annot=True,
            cmap='Oranges',
            vmin=0,
            vmax=1,
            xticklabels=list(self.model.classes_),
            yticklabels=list(self.model.classes_),
            ax=ax
        )
        ax.set_title('Normalized Confusion Matrix (rows sum up to 1)')
        ax.set_ylabel('True labels')
        ax.set_xlabel('Predicted labels')

    def plot_metrics_per_class(self, ax, on_set='validation', horizontal=True):
        """
        Plot precision, recall, and F1-score per class.

        Args:
            ax: Matplotlib axis to draw the plot on.
            on_set (str): Dataset to evaluate (either 'train' or 'validation').
            horizontal (bool): If True, plots horizontally; otherwise, vertically.
        """
        y = self.train_y if on_set == 'train' else self.val_y
        y_pred = self.train_y_pred if on_set == 'train' else self.val_y_pred
        index = sorted(y.unique().tolist())

        # when there is a zero division, precision is set to 0
        precision = pd.Series(precision_score(y, y_pred, average=None, zero_division=0, labels=index), index=index)
        recall = pd.Series(recall_score(y, y_pred, average=None, labels=index), index=index)
        f1 = pd.Series(f1_score(y, y_pred, average=None, labels=sorted(y.unique().tolist())), index=index)

        results = pd.DataFrame(pd.concat([precision, recall, f1], axis=1))
        results.columns = ['precision', 'recall', 'f1']

        if horizontal:
            sns.heatmap(results.T, vmin=0, vmax=1, cmap='Oranges', annot=True, ax=ax)
        else:
            sns.heatmap(results, vmin=0, vmax=1, cmap='Oranges', annot=True, ax=ax)


class DTClassificationEvaluation(ClassificationEvaluation):
    """
    A subclass for evaluating Decision Tree classification models and generating evaluation metrics.

    Args:
        model_name (str): The name of the classification model.
        model: The trained classification model.
        train_X (pd.DataFrame): The training data features.
        train_y (pd.Series): The training data labels.
        val_X (pd.DataFrame): The validation data features.
        val_y (pd.Series): The validation data labels.

    Methods:
        plot_feature_importances(ax, num_features=10, horizontal=False): Plot feature importances.
    """

    def __init__(self, model_name, model, train_X, train_y, val_X, val_y):
        """
        Initialize the DTClassificationEvaluation instance.

        Args:
            model_name (str): The name of the classification model.
            model: The trained classification model.
            train_X (pd.DataFrame): The training data features.
            train_y (pd.Series): The training data labels.
            val_X (pd.DataFrame): The validation data features.
            val_y (pd.Series): The validation data labels.
        """
        super().__init__(model_name, model, train_X, train_y, val_X, val_y)

    def plot_feature_importances(self, ax, num_features=10, horizontal=False):
        """
        Plot feature importances.

        Args:
            ax: Matplotlib axis to draw the plot on.
            num_features (int): Number of features to plot (default is 10).
            horizontal (bool): If True, plots horizontally; otherwise, vertically.
        """
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        names = [str(self.train_X.columns[i]) for i in indices][:num_features]
        if horizontal:
            sns.barplot(y=importances[indices][:num_features], x=names, ax=ax)
        else:
            sns.barplot(x=importances[indices][:num_features], y=names, ax=ax)
