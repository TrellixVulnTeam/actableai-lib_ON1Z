import pandas as pd

from actableai.timeseries.util import handle_datetime_column

simple_dt = pd.to_datetime("18-08-2021")
hard_dt = pd.to_datetime("01-08-2021")

dt_formats = ["%d-%m-%Y", "%d/%m/%Y", "%m-%d-%Y", "%m/%d/%Y", "%d-%m-%Y %H:%M:%S"]


class TestParseDatetime:
    def test_simple_datetime(self):
        dt_series = pd.Series([simple_dt.strftime(dt_formats[0])])
        r, col_type = handle_datetime_column(dt_series)
        assert pd.api.types.is_datetime64_ns_dtype(r) == True
        assert col_type == "datetime"
        assert r.isnull().sum() == 0

    def test_mixed_datetime(self):
        dt_series = pd.Series([simple_dt.strftime(x) for x in dt_formats])
        r, col_type = handle_datetime_column(dt_series)
        assert pd.api.types.is_datetime64_ns_dtype(r) == True
        assert col_type == "datetime"
        assert r.isnull().sum() == 0

    def test_multi_format_datetime(self):
        dt_series = pd.Series(
            [hard_dt.strftime(dt_formats[2]), simple_dt.strftime(dt_formats[0])]
            + [simple_dt.strftime(x) for x in dt_formats]
        )
        r, col_type = handle_datetime_column(dt_series)
        assert pd.api.types.is_datetime64_ns_dtype(r) == True
        assert col_type == "datetime"
        assert r.iloc[0] == pd.to_datetime(
            hard_dt.strftime(dt_formats[2]), format=dt_formats[0]
        )
        assert r.isnull().sum() == 0

    def test_non_datetime(self):
        non_dt_series = pd.Series(["a", "b", "c"])
        r, col_type = handle_datetime_column(non_dt_series)
        assert (r == non_dt_series).all()
        assert col_type == "others"
        assert r.isnull().sum() == 0

    def test_year_day_month_format(self):
        from actableai.timeseries.util import handle_datetime_column

        rng = pd.Series(pd.date_range("2015-01-01", periods=24, freq="MS"))
        ydm_series = pd.to_datetime(rng, format="%Y-%d-%m").astype(str)
        r, col_type = handle_datetime_column(ydm_series)
        assert (r == rng).all()
        assert pd.api.types.is_datetime64_ns_dtype(r) == True
        assert col_type == "datetime"

    def test_datetime_contain_milliseconds(self):
        from actableai.timeseries.util import handle_datetime_column

        rng = pd.Series(pd.date_range("2015-01-01", periods=24, freq="U"))
        r, col_type = handle_datetime_column(rng)
        assert (r == rng).all()
        assert pd.api.types.is_datetime64_ns_dtype(r) == True
        assert col_type == "datetime"
