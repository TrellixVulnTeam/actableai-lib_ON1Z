import pandas as pd
from typing import Union, Text

LoaderType = Union[Text, pd.DataFrame]


def load_data(d: LoaderType) -> pd.DataFrame:
    if isinstance(d, str):
        df = pd.read_csv(d)
    elif isinstance(d, pd.DataFrame):
        df = d
    else:
        raise TypeError(f"unsupported type {type(d)}")

    return df
