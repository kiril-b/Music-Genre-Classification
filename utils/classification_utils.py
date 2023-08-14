import numpy as np
import pandas as pd
import seaborn as sns
import IPython.display as ipd

from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler


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


class DatasetsWrapper():
    """A class to manage and preprocess datasets.

    This class wraps multiple datasets and provides methods for scaling and retrieving dataset subsets.

    Args:
        datasets (dict): A dictionary containing dataset names as keys and tuple (train, val, test) sets as values.

    Attributes:
        data_dict (dict): A dictionary holding the datasets with their corresponding subsets.

    Methods:
        scale_datasets(scaler): Scale all datasets using the given scaler.
        get_dataset(name, subset): Retrieve a specific subset of a dataset.
        get_shapes(): Return a DataFrame of shapes of all dataset subsets.

    """
    def __init__(self, datasets):
        self.data_dict = datasets
    
    def scale_datasets(self, scaler):
        """Scale all datasets using the provided scaler.

        Args:
            scaler: A scaler object to scale the datasets in-place.

        """
        for name, datasets in self.data_dict.items():
            train, val, test = datasets

            train = pd.DataFrame(scaler.fit_transform(train), index=train.index, columns=train.columns)
            val   = pd.DataFrame(scaler.transform(val), index=val.index, columns=val.columns)
            test  = pd.DataFrame(scaler.transform(test), index=test.index, columns=test.columns)

            self.data_dict[name] = (train, val, test)
    
    def get_dataset(self, name, subset):
        """Retrieve a specific subset of a dataset.

        Args:
            name (str): The name of the dataset.
            subset (str): The subset to retrieve ('train', 'val', or 'test').

        Returns:
            pd.DataFrame: The requested dataset subset.

        Raises:
            ValueError: If the subset value is not 'train', 'val', or 'test'.

        """
        dataset = self.data_dict[name]
        if subset == 'train':
            return dataset[0]
        elif subset == 'val':
            return dataset[1]
        elif subset == 'test':
            return dataset[2]
        else:
            raise ValueError('Invalid value for "subset". Allowed values: train, val, test')
    
    def get_shapes(self):
        """Return a DataFrame of shapes of all dataset subsets.

        Returns:
            pd.DataFrame: A DataFrame containing shapes of train, val, and test subsets.

        """
        shapes = [[_set.shape for _set in sets] for sets in self.data_dict.values()]
        index = self.data_dict.keys()
        return pd.DataFrame(shapes, index=index, columns=['train', 'val', 'test'])


class Classifier:
    """A class to represent a machine learning classifier.

    This class encapsulates a classifier model, its training and evaluation data, and provides methods
    to fit the classifier and evaluate its performance.

    Args:
        name (str): The name of the classifier.
        train_X (pd.DataFrame): The training features.
        train_y (pd.Series): The training labels.
        test_X (pd.DataFrame): The test/validation features.
        test_y (pd.Series): The test/validation labels.
        classifier: The classifier model.

    Attributes:
        name (str): The name of the classifier.
        train_X (pd.DataFrame): The training features.
        train_y (pd.Series): The training labels.
        test_X (pd.DataFrame): The test/validation features.
        test_y (pd.Series): The test/validation labels.
        classifier: The classifier model.
        evaluation: The ClassificationEvaluation object for storing evaluation results.

    Methods:
        fit_classifier(): Fit the classifier using the training data.
        evaluate_classifier(): Evaluate the classifier's performance on the test/validation data.

    """
    def __init__(self, name, train_X, train_y, test_X, test_y, classifier):
        self.name = name
        self.train_X = train_X
        self.train_y = train_y
        self.test_X = test_X
        self.test_y = test_y
        self.classifier = classifier
        self.evaluation = None
    
    def fit_classifier(self):
        """Fit the classifier using the training data."""
        self.classifier.fit(self.train_X, self.train_y)


    def evaluate_classifier(self):
        """Evaluate the classifier's performance on the test/validation data.

        Returns:
            dict: Evaluation scores.

        """
        if self.evaluation is None:
                self.evaluation = ClassificationEvaluation(
                    model_name=self.name,
                    model=self.classifier,
                    train_X=self.train_X, train_y=self.train_y,
                    val_X=self.test_X, val_y=self.test_y
                )
        return self.evaluation.get_scores(on_sets=['train', 'validation'])
    

class ClassifiersCollection:
    """A collection of classifiers for joint management and evaluation.

    This class manages a collection of Classifier instances, providing methods to fit and evaluate them.

    Args:
        classifiers (list): List of Classifier instances.

    Attributes:
        classifiers (list): List of Classifier instances.

    Methods:
        fit_classifiers(): Fit all classifiers in the collection.
        evaluate_classifiers(): Evaluate the performance of all classifiers.
        get_classifier(name): Get a specific classifier by name.
        free_memory(): Delete classifiers to free memory.

    """
    def __init__(self, classifiers):
        self.classifiers = classifiers
    
    def fit_classifiers(self):
        """Fit all classifiers in the collection."""
        print('Fitting classifiers...')
        for classifier in tqdm(self.classifiers):
            classifier.fit_classifier()
        print('Done')
    
    def evaluate_classifiers(self):
        """Evaluate the performance of all classifiers."""
        for classifier in self.classifiers:
            print(classifier.name)
            ipd.display(classifier.evaluate_classifier())
    
    def get_classifier(self, name):
        """Get a specific classifier by name.

        Args:
            name (str): The name of the classifier to retrieve.

        Returns:
            Classifier: The requested classifier instance, or None if not found.

        """
        return next(filter(lambda c: c.name == name, self.classifiers), None)
    
    def free_memory(self):
        """Delete classifiers to free memory."""
        del self.classifiers