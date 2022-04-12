from pandas import date_range
import pandas as pd
import pytest
from actableai.timeseries.util import handle_datetime_column


@pytest.mark.parametrize(
    "start_date", ["2015-09-08", "18:00", "18:30:25", "2020-01-02 05:00:00+02:00"]
)
@pytest.mark.parametrize("freq", ["T", "H", "S", "Y", "us"])
def test_handle_datetime_colum_pd_datetime(start_date: str, freq):
    date_range = pd.Series(pd.date_range(start_date, periods=10, freq=freq))
    parsed_dt, dtype = handle_datetime_column(date_range)

    assert parsed_dt is not None
    assert dtype == "datetime"


@pytest.mark.parametrize("start_date", ["2015-09-08", "2020-01-02 05:00:00+02:00"])
@pytest.mark.parametrize("freq", ["T", "H", "S", "Y", "us"])
def test_handle_datetime_colum_str(start_date: str, freq):
    date_range: pd.Series = pd.Series(
        pd.date_range(start_date, periods=10, freq=freq)
    ).astype(str)
    parsed_dt, dtype = handle_datetime_column(date_range)

    assert parsed_dt is not None
    assert dtype == "datetime"


def test_handle_datetime_column_mixed_hour():
    date_range: pd.Series = pd.Series(
        pd.date_range("18:00", periods=10, freq="H")
    ).astype(str)
    parsed_dt, dtype = handle_datetime_column(date_range)

    assert parsed_dt is not None
    assert dtype == "datetime"


@pytest.mark.parametrize("freq", ["T", "S", "Y", "us"])
def test_handle_datetime_column_hour_multi_freq(freq):
    date_range: pd.Series = pd.Series(
        pd.date_range("18:00", periods=10, freq=freq)
    ).astype(str)
    parsed_dt, dtype = handle_datetime_column(date_range)

    assert parsed_dt is not None
    assert dtype == "datetime"


def test_handle_datetime_column_tstamp_mixed():
    date_range: pd.Series = pd.Series(
        pd.date_range("18:30:25", periods=10, freq="H")
    ).astype(str)
    parsed_dt, dtype = handle_datetime_column(date_range)

    assert parsed_dt is not None
    assert dtype == "datetime"


@pytest.mark.parametrize("freq", ["T", "S", "Y", "us"])
def test_handle_datetime_column_tstamp_multi_freq(freq):
    date_range: pd.Series = pd.Series(
        pd.date_range("18:30:25", periods=10, freq=freq)
    ).astype(str)
    parsed_dt, dtype = handle_datetime_column(date_range)

    assert parsed_dt is not None
    assert dtype == "datetime"


def test_handle_datetime_column_dotted():
    date_range = pd.Series(["{}.{}.20{:0>2}".format(i, i, i) for i in range(1, 10)])
    parsed_dt, dtype = handle_datetime_column(date_range)

    assert parsed_dt is not None
    assert dtype == "datetime"


def test_handle_datetime_column_nan_values_threshold():
    date_range = pd.Series(["{}.{}.20{:0>2}".format(i, i, i) for i in range(1, 10)])
    date_range[0:3] = ["abc" for _ in range(3)]
    parsed_dt, dtype = handle_datetime_column(date_range)

    assert parsed_dt is not None
    assert dtype == "datetime"
    assert parsed_dt.isna().sum() == 3


def test_handle_datetime_column_nan_values_no_threshold():
    date_range = pd.Series(["{}.{}.20{:0>2}".format(i, i, i) for i in range(1, 10)])
    date_range[0:7] = ["abc" for _ in range(7)]
    parsed_dt, dtype = handle_datetime_column(date_range)

    assert parsed_dt is not None
    assert dtype == "others"
