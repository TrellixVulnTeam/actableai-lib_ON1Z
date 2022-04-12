import pandas as pd
from pandas.api.types import is_datetime64tz_dtype
from actableai.utils.sanitize import sanitize_timezone


def test_sanitize():
    df = pd.DataFrame(
        {
            "x": pd.date_range(
                start="27/03/1997", periods=120, freq="S", tz="US/Eastern"
            ),
            "y": pd.date_range(
                start="27/03/1997 12:00", periods=120, freq="S", tz="Asia/Singapore"
            ),
            "z": pd.date_range(
                start="27/03/1997 23:30:30", periods=120, freq="S", tz="US/Eastern"
            ),
        }
    )

    df = sanitize_timezone(df)
    assert df.apply(is_datetime64tz_dtype).all() == False
