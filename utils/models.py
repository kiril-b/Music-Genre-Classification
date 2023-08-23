import pandas as pd
from pydantic import BaseModel

class SplitDataset(BaseModel):
    """
    Represents a split dataset with features and labels.

    Attributes:
        X (pd.DataFrame): The feature data.
        y (Union[pd.Series, pd.DataFrame]): The label data, which can be either a Series or DataFrame.

    Config:
        arbitrary_types_allowed (bool): Whether arbitrary types are allowed for the attributes.
    """
    X: pd.DataFrame
    y: pd.Series

    class Config:
        arbitrary_types_allowed = True

class Dataset(BaseModel):
    """
    Represents a dataset with training, validation, and test splits.

    Attributes:
        name (str): The name of the dataset.
        train (SplitDataset): The training split.
        val (SplitDataset): The validation split.
        test (SplitDataset): The test split.
    """
    name: str
    train: SplitDataset
    val: SplitDataset
    test: SplitDataset