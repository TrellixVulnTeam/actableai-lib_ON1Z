from datetime import datetime

import pandas as pd
import pytest

from actableai.data_imputation.type_recon.helper import as_datetime


@pytest.mark.parametrize(
    "series, expect_series",
    [
        (
            pd.Series(data=["2021-02-01", "2021/02/01"]),
            pd.Series(
                data=[
                    datetime(year=2021, month=2, day=1),
                    datetime(year=2021, month=2, day=1),
                ]
            ),
        ),
        (
            pd.Series(data=["2021-2-1"]),
            pd.Series(data=[datetime(year=2021, month=2, day=1)]),
        ),
        (
            pd.Series(data=["01/02/2021"]),
            pd.Series(data=[datetime(year=2021, month=2, day=1)]),
        ),
        (
            pd.Series(data=["2021-02-01T01:10:23Z"]),
            pd.Series(
                data=[datetime(year=2021, month=2, day=1, hour=1, minute=10, second=23)]
            ),
        ),
        (
            pd.Series(data=["01/02/2021 01:10:23", "1/2/2021 1:10:23"]),
            pd.Series(
                data=[
                    datetime(year=2021, month=2, day=1, hour=1, minute=10, second=23),
                    datetime(year=2021, month=2, day=1, hour=1, minute=10, second=23),
                ]
            ),
        ),
        (
            pd.Series(data=["2021-02-01 01:10:23"]),
            pd.Series(
                data=[datetime(year=2021, month=2, day=1, hour=1, minute=10, second=23)]
            ),
        ),
        (
            pd.Series(data=["2021-02-01 01:10:23.022222"]),
            pd.Series(
                data=[
                    datetime(
                        year=2021,
                        month=2,
                        day=1,
                        hour=1,
                        minute=10,
                        second=23,
                        microsecond=22222,
                    )
                ]
            ),
        ),
    ],
)
def test_as_datetime(series, expect_series):
    assert as_datetime(series).equals(expect_series)
