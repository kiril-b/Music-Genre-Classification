import os
from typing import Union

from utils.models import Dataset, SplitDataset

import pandas as pd

from sklearn.base import TransformerMixin


class DatasetsWrapper:
    """A class for managing and manipulating multiple datasets.

    This class provides methods to scale datasets using a scaler, get shapes of datasets,
    and retrieve a specific dataset by name.

    Args:
        datasets (list[Dataset]): A list of Dataset objects to be managed.

    Attributes:
        data_dict (dict[str, Dataset]): A dictionary mapping dataset names to Dataset objects.

    """

    def __init__(self, datasets: list[Dataset]):
        """Initialize the DatasetsWrapper instance.

        Args:
            datasets (list[Dataset]): A list of Dataset objects to be managed.

        """
        self.data_dict = {dataset.name: dataset for dataset in datasets}

    def scale_datasets(
        self, scaler: TransformerMixin, exclude: list[str] | None = None
    ) -> None:
        """Scale the datasets using the provided scaler.

        This method scales the datasets using the specified scaler and updates
        the dataset objects with the scaled values.

        Args:
            scaler (TransformerMixin): A data scaler implementing the TransformerMixin interface.
            exclude (list[str], optional): A list of dataset names to exclude from scaling. Defaults to None.

        """
        for name, dataset in self.data_dict.items():
            if exclude is not None and name in exclude:
                continue

            train, val, test = dataset.train, dataset.val, dataset.test

            train.X = pd.DataFrame(
                scaler.fit_transform(train.X),
                index=train.X.index,
                columns=train.X.columns,
            )
            val.X = pd.DataFrame(
                scaler.transform(val.X), index=val.X.index, columns=val.X.columns
            )
            test.X = pd.DataFrame(
                scaler.transform(test.X), index=test.X.index, columns=test.X.columns
            )

    def encode_labels(self, encoder: TransformerMixin) -> None:
        """
        Encode labels in the datasets using the provided encoder.

        Args:
            encoder (TransformerMixin): The encoder to be used for label encoding.

        Returns:
            None
        """
        self.label_encoder = encoder

        for _, dataset in self.data_dict.items():
            train, val, test = dataset.train, dataset.val, dataset.test

            train.y = pd.Series(
                self.label_encoder.fit_transform(train.y), index=train.y.index
            )
            val.y = pd.Series(self.label_encoder.transform(val.y), index=val.y.index)
            test.y = pd.Series(self.label_encoder.transform(test.y), index=test.y.index)

    def reverse_encode_labels(self) -> None:
        """
        Reverse the label encoding in the datasets to their original values.

        Raises:
            Exception: If the labels were not encoded previously. Call "encode_labels" first.

        Returns:
            None
        """
        if self.label_encoder is None:
            raise Exception(
                'The lables were not encoded previously. Make sure to call "encode_labels" first'
            )

        for _, dataset in self.data_dict.items():
            train, val, test = dataset.train, dataset.val, dataset.test

            train.y = pd.Series(
                self.label_encoder.inverse_transform(train.y), index=train.y.index
            )
            val.y = pd.Series(
                self.label_encoder.inverse_transform(val.y), index=val.y.index
            )
            test.y = pd.Series(
                self.label_encoder.inverse_transform(test.y), index=test.y.index
            )

    def get_shapes(self) -> pd.DataFrame:
        """Get the shapes of datasets.

        This method returns a DataFrame containing the shapes of train, validation,
        and test sets for each dataset.

        Returns:
            pd.DataFrame: A DataFrame with dataset shapes, indexed by dataset names.

        """
        shapes = [
            [
                dataset.train.X.shape,
                dataset.val.X.shape,
                dataset.test.X.shape,
                dataset.train.y.shape,
                dataset.val.y.shape,
                dataset.test.y.shape,
            ]
            for dataset in self.data_dict.values()
        ]
        index = self.data_dict.keys()

        return pd.DataFrame(
            shapes,
            index=index,
            columns=["train_X", "val_X", "test_X", "train_y", "val_y", "test_y"],
        )

    def get_dataset(self, name: str) -> Dataset:
        """Get a specific dataset by name.

        Args:
            name (str): The name of the dataset to retrieve.

        Returns:
            Dataset: The Dataset object corresponding to the specified name.

        """
        return self.data_dict[name]


def create_dataset(
    name: str,
    train_X: pd.DataFrame,
    train_y: Union[pd.DataFrame, pd.Series],
    val_X: pd.DataFrame,
    val_y: Union[pd.DataFrame, pd.Series],
    test_X: pd.DataFrame,
    test_y: Union[pd.DataFrame, pd.Series],
) -> Dataset:
    """
    Create a Dataset object with train, validation, and test data splits.

    Args:
        name (str): Name of the dataset.
        train_X (pd.DataFrame): Training features.
        train_y (Union[pd.DataFrame, pd.Series]): Training labels.
        val_X (pd.DataFrame): Validation features.
        val_y (Union[pd.DataFrame, pd.Series]): Validation labels.
        test_X (pd.DataFrame): Test features.
        test_y (Union[pd.DataFrame, pd.Series]): Test labels.

    Returns:
        Dataset: A Dataset object containing the splits.
    """
    return Dataset(
        name=name,
        train=SplitDataset(X=train_X, y=train_y),
        val=SplitDataset(X=val_X, y=val_y),
        test=SplitDataset(X=test_X, y=test_y),
    )


def load_datasets() -> list[Dataset]:
    """
    Load datasets from files and create Dataset objects.

    Returns:
        dict[str, Dataset]: A dictionary containing Dataset objects for various dataset variations.
    """
    train_set_original = pd.read_csv(
        os.getenv("TRAIN_SET_ORIGINAL_PATH"), index_col=0, header=[0, 1, 2]
    )
    train_set_modified = pd.read_csv(
        os.getenv("TRAIN_SET_MODIFIED_PATH"), index_col=0, header=[0, 1, 2]
    )
    val_set = pd.read_csv(
        os.getenv("VALIDATION_SET_PATH"), index_col=0, header=[0, 1, 2]
    )
    test_set = pd.read_csv(os.getenv("TEST_SET_PATH"), index_col=0, header=[0, 1, 2])

    genre_column = ("genre", "Unnamed: 253_level_1", "Unnamed: 253_level_2")
    train_original_y = train_set_original[genre_column]
    train_modified_y = train_set_modified[genre_column]
    val_y = val_set[genre_column]
    test_y = test_set[genre_column]

    train_spectral_original_X = pd.read_csv(
        os.getenv("train_spectral_original_X"), index_col=0, header=[0, 1, 2]
    )
    train_spectral_modified_X = pd.read_csv(
        os.getenv("train_spectral_modified_X"), index_col=0, header=[0, 1, 2]
    )
    val_spectral_X = pd.read_csv(
        os.getenv("val_spectral_X"), index_col=0, header=[0, 1, 2]
    )
    test_spectral_X = pd.read_csv(
        os.getenv("test_spectral_X"), index_col=0, header=[0, 1, 2]
    )

    train_all_original_X = pd.read_csv(
        os.getenv("train_all_original_X"), index_col=0, header=[0, 1, 2]
    )
    train_all_modified_X = pd.read_csv(
        os.getenv("train_all_modified_X"), index_col=0, header=[0, 1, 2]
    )
    val_all_X = pd.read_csv(os.getenv("val_all_X"), index_col=0, header=[0, 1, 2])
    test_all_X = pd.read_csv(os.getenv("test_all_X"), index_col=0, header=[0, 1, 2])

    train_original_pca_X = pd.read_csv(os.getenv("train_original_pca_X"), index_col=0)
    val_original_pca_X = pd.read_csv(os.getenv("val_original_pca_X"), index_col=0)
    test_original_pca_X = pd.read_csv(os.getenv("test_original_pca_X"), index_col=0)
    train_modified_pca_X = pd.read_csv(os.getenv("train_modified_pca_X"), index_col=0)
    val_modified_pca_X = pd.read_csv(os.getenv("val_modified_pca_X"), index_col=0)
    test_modified_pca_X = pd.read_csv(os.getenv("test_modified_pca_X"), index_col=0)

    train_original_umap_X = pd.read_csv(os.getenv("train_original_umap_X"), index_col=0)
    val_original_umap_X = pd.read_csv(os.getenv("val_original_umap_X"), index_col=0)
    test_original_umap_X = pd.read_csv(os.getenv("test_original_umap_X"), index_col=0)
    train_modified_umap_X = pd.read_csv(os.getenv("train_modified_umap_X"), index_col=0)
    val_modified_umap_X = pd.read_csv(os.getenv("val_modified_umap_X"), index_col=0)
    test_modified_umap_X = pd.read_csv(os.getenv("test_modified_umap_X"), index_col=0)

    spectral_original_dataset = create_dataset(
        "spectral_original",
        train_spectral_original_X,
        train_original_y,
        val_spectral_X,
        val_y,
        test_spectral_X,
        test_y,
    )

    all_original_dataset = create_dataset(
        "all_original",
        train_all_original_X,
        train_original_y,
        val_all_X,
        val_y,
        test_all_X,
        test_y,
    )

    pca_original_dataset = create_dataset(
        "pca_original",
        train_original_pca_X,
        train_original_y,
        val_original_pca_X,
        val_y,
        test_original_pca_X,
        test_y,
    )

    umap_original_dataset = create_dataset(
        "umap_original",
        train_original_umap_X,
        train_original_y,
        val_original_umap_X,
        val_y,
        test_original_umap_X,
        test_y,
    )

    spectral_modified_dataset = create_dataset(
        "spectral_modified",
        train_spectral_modified_X,
        train_modified_y,
        val_spectral_X,
        val_y,
        test_spectral_X,
        test_y,
    )

    all_modified_dataset = create_dataset(
        "all_modified",
        train_all_modified_X,
        train_modified_y,
        val_all_X,
        val_y,
        test_all_X,
        test_y,
    )

    pca_modified_dataset = create_dataset(
        "pca_modified",
        train_modified_pca_X,
        train_modified_y,
        val_modified_pca_X,
        val_y,
        test_modified_pca_X,
        test_y,
    )

    umap_modified_dataset = create_dataset(
        "umap_modified",
        train_modified_umap_X,
        train_modified_y,
        val_modified_umap_X,
        val_y,
        test_modified_umap_X,
        test_y,
    )

    return [
        spectral_original_dataset,
        all_original_dataset,
        pca_original_dataset,
        umap_original_dataset,
        spectral_modified_dataset,
        all_modified_dataset,
        pca_modified_dataset,
        umap_modified_dataset,
    ]
