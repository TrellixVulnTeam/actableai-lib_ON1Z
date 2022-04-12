import pandas as pd

from actableai.utils import get_type_special_no_ag


all_dataset_features = [
    "dataset_type",
    "dataset_row_count",
    "dataset_categorical_column_short_count",
    "dataset_categorical_column_long_count",
    "dataset_text_column_unique_short_count",
    "dataset_text_column_unique_long_count",
    "dataset_numbers_int_column_count",
    "dataset_numbers_float_column_count",
    "dataset_categorical_column_count",
    "dataset_text_column_count",
    "dataset_numbers_column_count",
    "dataset_column_count",
]


def _is_column_short(series: pd.Series, max_len: int = 6) -> bool:
    """
    Check if a Pandas Series is a short column

    Parameters
    ----------
    series:
        The Pandas Series to check
    max_len:
        For the function to return True, all words of each row must have a maximum length of `max_len`

    Returns
    -------
    True if the column is considered short
    """
    series_word_count = series.astype("string").str.count(" ") + 1
    series_len = series.astype("string").str.len()

    # Returns true if all the rows contain less than `max_len` times the number of character
    # + the number of word minus one for the spaces
    return (series_len < (series_word_count * (max_len + 1) - 1)).all()


def _is_column_float(series: pd.Series) -> bool:
    """
    Check if a Pandas Series contains only floating point numbers

    Parameters
    ----------
    series:
        The Pandas Series to check

    Returns
    -------
    True if the column is float
    """
    return series.dtypes == float


def extract_dataset_features(df_dataset: pd.DataFrame, prefix: str = "") -> dict:
    """
    Extract features from a dataset

    Parameters
    ----------
    df_dataset:
        The Pandas DataFrame dataset to extract the features from
    prefix:
        Prefix to add to all the features

    Returns
    -------
    The features extracted
    """
    dataset_categorical_column_short_count = 0
    dataset_categorical_column_long_count = 0
    dataset_text_column_unique_short_count = 0
    dataset_text_column_unique_long_count = 0
    dataset_numbers_int_column_count = 0
    dataset_numbers_float_column_count = 0

    for column in df_dataset.columns:
        # Get the type of the column
        column_type = get_type_special_no_ag(df_dataset[column])
        if column_type != "category" and column_type != "text":
            column_type = "numbers"

        if column_type == "category":
            if _is_column_short(df_dataset[column]):
                dataset_categorical_column_short_count += 1
            else:
                dataset_categorical_column_long_count += 1
        elif column_type == "text":
            if _is_column_short(df_dataset[column]):
                dataset_text_column_unique_short_count += 1
            else:
                dataset_text_column_unique_long_count += 1
        elif column_type == "numbers":
            if _is_column_float(df_dataset[column]):
                dataset_numbers_float_column_count += 1
            else:
                dataset_numbers_int_column_count += 1
        else:
            raise Exception("This should not happen")

    dataset_categorical_column_count = (
        dataset_categorical_column_short_count + dataset_categorical_column_long_count
    )
    dataset_text_column_count = (
        dataset_text_column_unique_short_count + dataset_text_column_unique_long_count
    )
    dataset_numbers_column_count = (
        dataset_numbers_int_column_count + dataset_numbers_float_column_count
    )
    dataset_column_count = len(df_dataset.columns)
    dataset_row_count = len(df_dataset)

    # Infer the dataset type
    dataset_type = "not_categorical"
    if dataset_categorical_column_count == len(df_dataset.columns):
        dataset_type = "categorical"
    elif dataset_categorical_column_count > 0:
        dataset_type = "mixed"

    return {
        f"{prefix}dataset_type": dataset_type,
        f"{prefix}dataset_row_count": dataset_row_count,
        f"{prefix}dataset_categorical_column_short_count": dataset_categorical_column_short_count,
        f"{prefix}dataset_categorical_column_long_count": dataset_categorical_column_long_count,
        f"{prefix}dataset_text_column_unique_short_count": dataset_text_column_unique_short_count,
        f"{prefix}dataset_text_column_unique_long_count": dataset_text_column_unique_long_count,
        f"{prefix}dataset_numbers_int_column_count": dataset_numbers_int_column_count,
        f"{prefix}dataset_numbers_float_column_count": dataset_numbers_float_column_count,
        f"{prefix}dataset_categorical_column_count": dataset_categorical_column_count,
        f"{prefix}dataset_text_column_count": dataset_text_column_count,
        f"{prefix}dataset_numbers_column_count": dataset_numbers_column_count,
        f"{prefix}dataset_column_count": dataset_column_count,
    }
