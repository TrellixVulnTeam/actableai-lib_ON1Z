import pandas as pd
from pandas.api.types import is_datetime64tz_dtype


def sanitize_timezone(df: pd.DataFrame) -> pd.DataFrame:
    """Sanitize TimeZone from DataFrame

    Args:
        df (pd.DataFrame): Original DataFrame

    Returns:
        pd.DataFrame: Sanitized Dataframe
    """
    # Convert any 'datetime with timezone' to 'datetime without timezone on UTC'
    tz_columns = df.apply(is_datetime64tz_dtype)
    df.loc[:, tz_columns] = df.loc[:, tz_columns].apply(lambda x: x.dt.tz_convert(None))
    return df
