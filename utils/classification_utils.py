import numpy as np
import pandas as pd
import seaborn as sns
import IPython.display as ipd
import matplotlib.colors as mcolors
import matplotlib.axes._axes as Axes
from sklearn.base import ClassifierMixin

from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler

from utils.models import Dataset


class ClassificationEvaluation:
    """
    A class for evaluating classification models using various metrics.

    Args:
        dataset (Dataset): The dataset containing train, validation, and test sets.
        classifier (ClassifierMixin): The classification model to be evaluated.
        encoder (ClassifierMixin, optional): The label encoder used for predictions. Default is None.

    Attributes:
        dataset (Dataset): The dataset used for evaluation.
        classifier (ClassifierMixin): The classification model being evaluated.
        label_encoder (ClassifierMixin, optional): The label encoder for predictions.
        predictions_made (bool): Indicates whether predictions have been made.

    Methods:
        make_predictions(): Make predictions using the classifier and store the results.
        get_scores(on_sets=None, average_type='weighted'): Calculate evaluation scores for specified datasets.
        plot_confusion_matrix(ax, on_set): Plot a normalized confusion matrix for the specified dataset.
        plot_metrics_per_class(ax, on_set, horizontal=True): Plot precision, recall, and F1-score per class.
    """

    def __init__(
        self,
        dataset: Dataset,
        classifier: ClassifierMixin,
        encoder: ClassifierMixin | None = None,
    ):
        """
        Initialize the ClassificationEvaluation instance.

        Args:
            dataset (Dataset): The dataset containing train, validation, and test sets.
            classifier (ClassifierMixin): The classification model to be evaluated.
            encoder (ClassifierMixin, optional): The label encoder used for predictions. Default is None.
        """
        self.dataset = dataset
        self.classifier = classifier
        self.label_encoder = encoder
        self.predictions_made = False

    def make_predictions(self) -> None:
        """
        Make predictions using the classifier and store the results.
        """
        train_y_pred = self.classifier.predict(self.dataset.train.X)
        val_y_pred = self.classifier.predict(self.dataset.val.X)
        test_y_pred = self.classifier.predict(self.dataset.test.X)

        # If the target columns was label encoded before fitting
        if self.label_encoder is not None and not isinstance(
            self.classifier.classes_[0], str
        ):
            train_y_pred = self.label_encoder.inverse_transform(train_y_pred)
            val_y_pred = self.label_encoder.inverse_transform(val_y_pred)
            test_y_pred = self.label_encoder.inverse_transform(test_y_pred)

        self.train_y_pred = pd.Series(train_y_pred, index=self.dataset.train.X.index)
        self.val_y_pred = pd.Series(val_y_pred, index=self.dataset.val.X.index)
        self.test_y_pred = pd.Series(test_y_pred, index=self.dataset.test.X.index)

        self.predictions_made = True

    def get_scores(
        self, on_sets: list[str] | None = None, average_type: str = "weighted"
    ) -> pd.DataFrame:
        """
        Calculate evaluation scores for specified datasets.

        Args:
            on_sets (list[str], optional): List of datasets to evaluate. Default is ['train', 'val', 'test'].
            average_type (str, optional): Type of averaging for precision, recall, and F1-score. Default is 'weighted'.

        Returns:
            pd.DataFrame: Evaluation scores (accuracy, precision, recall, F1-score) for specified datasets.
        """
        if not self.predictions_made:
            self.make_predictions()

        on_sets = ["train", "val", "test"] if on_sets is None else on_sets

        data = []
        index = [
            "accuracy",
            f"precision ({average_type})",
            f"recall ({average_type})",
            f"f1 ({average_type})",
        ]

        train_y, val_y, test_y = (
            self.dataset.train.y,
            self.dataset.val.y,
            self.dataset.test.y,
        )

        for dataset in on_sets:
            if dataset == "train":
                data.append(
                    [
                        accuracy_score(train_y, self.train_y_pred),
                        precision_score(
                            train_y,
                            self.train_y_pred,
                            average=average_type,
                            zero_division=0,
                        ),
                        recall_score(train_y, self.train_y_pred, average=average_type),
                        f1_score(train_y, self.train_y_pred, average=average_type),
                    ]
                )
            elif dataset == "val":
                data.append(
                    [
                        accuracy_score(val_y, self.val_y_pred),
                        precision_score(
                            val_y,
                            self.val_y_pred,
                            average=average_type,
                            zero_division=0,
                        ),
                        recall_score(val_y, self.val_y_pred, average=average_type),
                        f1_score(val_y, self.val_y_pred, average=average_type),
                    ]
                )
            elif dataset == "test":
                data.append(
                    [
                        accuracy_score(test_y, self.test_y_pred),
                        precision_score(
                            test_y,
                            self.test_y_pred,
                            average=average_type,
                            zero_division=0,
                        ),
                        recall_score(test_y, self.test_y_pred, average=average_type),
                        f1_score(test_y, self.test_y_pred, average=average_type),
                    ]
                )
            else:
                raise ValueError(
                    "on_sets may only contain values from the following: {'train', 'val', 'test'}"
                )

        return pd.DataFrame(data, columns=index, index=on_sets).T

    def plot_confusion_matrix(self, ax: Axes, on_set: str) -> None:
        """
        Plot a normalized confusion matrix for the specified dataset.

        Args:
            ax (Axes): Matplotlib Axes object for plotting.
            on_set (str): Dataset to plot the confusion matrix for.
        """
        if not self.predictions_made:
            self.make_predictions()

        if on_set not in ["train", "val", "test"]:
            raise ValueError(
                "on_sets may only be one of the following: ['train', 'val', 'test']'"
            )

        y, y_pred = None, None
        if on_set == "train":
            y, y_pred = self.dataset.train.y, self.train_y_pred
        elif on_set == "val":
            y, y_pred = self.dataset.val.y, self.val_y_pred
        else:
            y, y_pred = self.dataset.test.y, self.test_y_pred

        cm = confusion_matrix(y, y_pred, normalize="true")
        sns.heatmap(
            cm,
            annot=True,
            cmap=mcolors.LinearSegmentedColormap.from_list(
                "custom_colormap", ["white", "#34d4a1"]
            ),
            vmin=0,
            vmax=1,
            # xticklabels=list(self.classifier.classes_),
            # yticklabels=list(self.classifier.classes_),
            xticklabels=sorted(self.dataset.test.y.unique().tolist()),
            yticklabels=sorted(self.dataset.test.y.unique().tolist()),
            ax=ax,
        )
        ax.set_title(f"Normalized Confusion Matrix on {on_set} set (rows sum up to 1)")
        ax.set_ylabel("True labels")
        ax.set_xlabel("Predicted labels")

    def plot_metrics_per_class(
        self, ax: Axes, on_set: str, horizontal: bool = True
    ) -> None:
        """
        Plot precision, recall, and F1-score per class for the specified dataset.

        Args:
            ax (Axes): Matplotlib Axes object for plotting.
            on_set (str): Dataset to plot the metrics for.
            horizontal (bool, optional): If True, metrics are plotted horizontally. Default is True.
        """
        if not self.predictions_made:
            self.make_predictions()

        if on_set not in ["train", "val", "test"]:
            raise ValueError(
                "on_sets may only be one of the following: ['train', 'val', 'test']'"
            )

        y, y_pred = None, None
        if on_set == "train":
            y, y_pred = self.dataset.train.y, self.train_y_pred
        elif on_set == "val":
            y, y_pred = self.dataset.val.y, self.val_y_pred
        else:
            y, y_pred = self.dataset.test.y, self.test_y_pred

        index = sorted(y.unique().tolist())

        # when there is a zero division, precision is set to 0
        precision = pd.Series(
            precision_score(y, y_pred, average=None, zero_division=0, labels=index),
            index=index,
        )
        recall = pd.Series(
            recall_score(y, y_pred, average=None, labels=index), index=index
        )
        f1 = pd.Series(f1_score(y, y_pred, average=None, labels=index), index=index)

        results = pd.DataFrame(pd.concat([precision, recall, f1], axis=1))
        results.columns = ["precision", "recall", "f1"]

        ax.set_title(f"Metrics for {on_set} set")
        if horizontal:
            results = results.T

        sns.heatmap(
            results,
            vmin=0,
            vmax=1,
            cmap=mcolors.LinearSegmentedColormap.from_list(
                "custom_colormap", ["white", "#34d4a1"]
            ),
            annot=True,
            ax=ax,
        )


class DTClassificationEvaluation(ClassificationEvaluation):
    """A subclass of ClassificationEvaluation specifically for Decision Tree classifiers.

    Args:
        dataset (Dataset): The dataset containing train, validation, and test data.
        classifier (ClassifierMixin): The trained decision tree classifier model.

    Methods:
        plot_feature_importances: Plot feature importances for a decision tree classifier.

    """

    def __init__(
        self, dataset: Dataset, classifier: ClassifierMixin, encoder: ClassifierMixin
    ):
        super().__init__(dataset, classifier, encoder)

    def plot_feature_importances(
        self, ax: Axes, num_features: int = 10, horizontal: bool = False
    ) -> None:
        """
        Plot feature importances for a decision tree classifier.

        Parameters:
            ax (Axes): Matplotlib Axes object to plot on.
            num_features (int, optional): Number of top features to plot. Default is 10.
            horizontal (bool, optional): Whether to plot horizontally. Default is False.

        This method plots the top feature importances for a decision tree classifier using seaborn barplot.
        """
        importances = self.classifier.feature_importances_
        indices = np.argsort(importances)[::-1]
        names = [str(self.dataset.train.X.columns[i]) for i in indices][:num_features]
        if horizontal:
            sns.barplot(
                y=importances[indices][:num_features], x=names, ax=ax, color="coral"
            )
        else:
            sns.barplot(
                x=importances[indices][:num_features], y=names, ax=ax, color="coral"
            )


class Classifier:
    """A class representing a machine learning classifier.

    This class encapsulates the process of training and evaluating a classifier on a given dataset.

    Args:
        dataset (Dataset): The dataset to be used for training and evaluation.
        classifier (ClassifierMixin): The classifier model to be used.

    Attributes:
        name (str): The name of the classifier
        dataset (Dataset): The dataset used for training and evaluation.
        classifier (ClassifierMixin): The classifier model.
        evaluation (ClassificationEvaluation): The evaluation results container.

    Methods:
        fit_classifier: Fits the classifier on the training data.
        evaluate_classifier: Evaluates the classifier on the train and validation sets.

    """

    def __init__(
        self,
        dataset: Dataset,
        classifier: ClassifierMixin,
        evaluation: ClassificationEvaluation,
    ):
        self.name = dataset.name
        self.dataset = dataset
        self.classifier = classifier
        self.evaluation = evaluation

    def fit_classifier(self) -> None:
        """Fits the classifier on the training data."""
        self.classifier.fit(self.dataset.train.X, self.dataset.train.y)

    def get_clf_evaluation(self) -> ClassificationEvaluation:
        """
        Retrieve a ClassificationEvaluation instance for the classifier's performance on a dataset.

        Returns:
            ClassificationEvaluation: A ClassificationEvaluation instance containing evaluation metrics.
        """
        return self.evaluation


class ClassifiersCollection:
    """A collection of machine learning classifiers for batch processing.

    This class allows fitting and evaluating multiple classifiers in batch.

    Args:
        classifiers (list[Classifier]): List of Classifier instances to be managed.

    Attributes:
        classifiers (list[Classifier]): List of Classifier instances.

    Methods:
        fit_classifiers: Fits all classifiers in the collection.
        evaluate_classifiers: Evaluates and displays the performance of all classifiers.
        get_classifier: Retrieves a specific classifier by name.
        get_all_classifiers: Returns all of the classifiers.

    """

    def __init__(self, classifiers: list[Classifier]):
        self.classifiers = classifiers

    def fit_classifiers(self) -> None:
        """Fits all classifiers in the collection."""
        print("Fitting classifiers...")
        for classifier in tqdm(self.classifiers):
            classifier.fit_classifier()
        print("Done")

    def evaluate_classifiers(self) -> pd.DataFrame:
        """
        Evaluate multiple classifiers and return a DataFrame with their evaluation scores.

        The function iterates through each classifier, retrieves their evaluation scores for
        training, validation, and test sets, and creates a DataFrame with multi-level columns
        indicating the classifier and evaluation metric.

        Returns:
            pd.DataFrame: A DataFrame containing evaluation scores for each classifier.
        """
        dfs = list()
        for classifier in self.classifiers:
            df_scores = classifier.get_clf_evaluation().get_scores(
                on_sets=["train", "val", "test"]
            )
            new_columns = [(classifier.name, col) for col in df_scores.columns]
            df_scores.columns = pd.MultiIndex.from_tuples(new_columns)
            dfs.append(df_scores)

        return pd.concat(dfs, axis=1)

    def get_classifier(self, name: str) -> Classifier:
        """Retrieves a classifier by its name.

        Args:
            name (str): The name of the classifier to retrieve.

        Returns:
            Classifier: The requested Classifier instance, or None if not found.

        Raises:
            ValueError: If a classifier with the given name does not exist.
        """
        classifier = next(filter(lambda c: c.name == name, self.classifiers), None)
        if classifier is None:
            raise ValueError(f"A classifier with name: {name} does not exist!")
        return classifier

    def get_all_classifiers(self) -> list[Classifier]:
        """Returns all of the classifiers that were passed in the constructor.

        Returns:
            list[Classifier]: the classifiers that were passed in the constructor
        """
        return self.classifiers


class ClassifierFactory:
    """
    A factory class to create instances of a classifier with associated evaluation based on parameters.

    This factory allows creating instances of a classifier with different types of evaluation objects,
    based on the type of classifier, providing a unified way to instantiate classifier objects.

    Args:
        dataset (Dataset): The dataset to be used for training and evaluation.
        classifier (ClassifierMixin): The classifier model to be used.
        encoder (ClassifierMixin): The encoder used to preprocess data for the classifier.
        tree_type_classifier (bool): A boolean indicating whether the classifier is a tree-based classifier.

    Returns:
        Classifier: An instance of Classifier with associated evaluation,
                    which could be ClassificationEvaluation or DTClassificationEvaluation,
                    based on the value of `tree_type_classifier`.
    """

    @staticmethod
    def create_instance(
        dataset: Dataset,
        classifier: ClassifierMixin,
        encoder: ClassifierMixin = None,
        tree_type_classifier: bool = False,
    ):
        """
        Create an instance of a classifier with associated evaluation.

        Args:
            dataset (Dataset): The dataset to be used for training and evaluation.
            classifier (ClassifierMixin): The classifier model to be used.
            encoder (ClassifierMixin): The encoder used to preprocess data for the classifier.
            tree_type_classifier (bool): A boolean indicating whether the classifier is a tree-based classifier.

        Returns:
            Classifier: An instance of Classifier with associated evaluation,
                        which could be ClassificationEvaluation or DTClassificationEvaluation,
                        based on the value of `tree_type_classifier`.
        """
        if tree_type_classifier:
            return Classifier(
                dataset=dataset,
                classifier=classifier,
                evaluation=DTClassificationEvaluation(dataset, classifier, encoder),
            )
        else:
            return Classifier(
                dataset=dataset,
                classifier=classifier,
                evaluation=ClassificationEvaluation(dataset, classifier, encoder),
            )
