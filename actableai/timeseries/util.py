import re
import json
from typing import Optional, Tuple
import numpy as np
import visions as v
import pandas as pd


def makeCorrectName(s):
    return s.replace(" ", "_")


def returnDataTable(df):
    exdata = json.loads(df.to_json(orient="table"))
    for item in exdata["data"]:
        if "index" in item:
            del item["index"]
    exdata["schema"]["fields"].pop(0)
    return exdata


def isCategory(column):
    if column in v.Float:
        return False
    if column in v.Integer:
        (dc,) = column.value_counts().shape
        if dc <= 5:
            return True
        else:
            return False
    return True


def findFredForGluon(freq):
    # For W:
    if re.findall("\d*W", freq):
        return re.findall("\d*W", freq)[0]
    elif re.findall("\d*M", freq):
        return re.findall("\d*M", freq)[0]

    return freq


def findFred(pd_date: pd.Series) -> Optional[str]:  ## Need to sorted before.
    if len(pd_date) < 3:
        return None
    pd_date = pd_date.sort_values()  # Sorting them ?
    freq = pd.infer_freq(pd_date)
    if freq == "MS":
        freq = "M"

    if freq:
        return freq

    infer_list = {}
    data_len = len(pd_date)
    for i in range(0, data_len - 3, 3):
        freq = pd.infer_freq(pd_date[i : i + 3])
        if freq is not None:
            if freq not in infer_list:
                infer_list[str(freq)] = 0
            infer_list[str(freq)] += 1
    if len(infer_list) == 0:
        return None
    most_freq = max(infer_list, key=lambda freq: infer_list[freq])
    return most_freq


def make_future_dataframe(periods, pd_date, freq, include_history=True):
    history_dates = pd.to_datetime(pd_date).sort_values()
    """Simulate the trend using the extrapolated generative model.
    Parameters
    ----------
    periods: Int number of periods to forecast forward.
    freq: Any valid frequency for pd.date_range, such as 'D' or 'M'.
    include_history: Boolean to include the historical dates in the data
        frame for predictions.
    Returns
    -------
    pd.Dataframe that extends forward from the end of self.history for the
    requested number of periods.
    """

    if history_dates is None:
        raise Exception("Model must be fit before this can be used.")
    last_date = history_dates.max()
    dates = pd.date_range(
        start=last_date,
        periods=periods + 1,  # An extra in case we include start
        freq=freq,
    )
    dates = dates[dates > last_date]  # Drop start if equals last_date
    dates = dates[:periods]  # Return correct number of periods

    if include_history:
        dates = np.concatenate((np.array(history_dates), dates))

    return pd.DataFrame({"ds": dates})


def minmax_scaler_fit_transform(df):
    from sklearn.preprocessing import MinMaxScaler

    x = df.values
    scaler = MinMaxScaler()
    x_scaled = scaler.fit_transform(x)
    x_scaled = pd.DataFrame(x_scaled)
    x_scaled.index = df.index

    return x_scaled, scaler


def minmax_scaler_transform(df, scaler):

    x = df.values
    x_scaled = scaler.transform(x)
    x_scaled = pd.DataFrame(x_scaled)
    x_scaled.index = df.index

    return x_scaled


def inverse_transform(normalized_fc, scaler):
    fc_samples = normalized_fc[0].samples
    fc_shape = fc_samples.shape
    fc_samples = scaler.inverse_transform(fc_samples.reshape(-1, fc_shape[2]))
    normalized_fc[0].samples = fc_samples.reshape(fc_shape)

    return normalized_fc


def get_satisfied_formats(row, unique_formats):
    satisfied_formats = []
    for format in unique_formats:
        try:
            pd.to_datetime(row, format=format)
            satisfied_formats.append(format)
        except:
            continue

    return satisfied_formats


def parse_datetime(dt_str, formats):
    for fm in formats:
        try:
            result = pd.to_datetime(dt_str, format=fm)
            return result
        except Exception:
            pass
    return None


def parse_by_format_with_valid_frequency(series, formats):
    for fm in formats:
        try:
            result = pd.to_datetime(series.astype(str), format=fm)
            if pd.infer_freq(result):
                return result
        except Exception:
            pass
    return pd.to_datetime(series, format=formats[0])


def handle_datetime_column(
    series: pd.Series, min_parsed_rate: float = 0.5
) -> Tuple[pd.Series, str]:
    from pandas._libs.tslibs.parsing import _guess_datetime_format

    parsed_rate_check = lambda x: x.isna().sum() >= min_parsed_rate * len(series)
    unique_formats = (
        pd.concat(
            [
                series.astype(str).apply(_guess_datetime_format, dayfirst=False),
                series.astype(str).apply(_guess_datetime_format, dayfirst=True),
            ]
        )
        .value_counts()
        .index.to_list()
    )
    if len(unique_formats) < 1:
        try:
            parsed_dt = pd.to_datetime(series, errors="coerce")
            if parsed_rate_check(parsed_dt):
                return series, "others"
            return parsed_dt, "datetime"
        except Exception:
            pass

        column_dtype = "others"
        return series, column_dtype
    else:
        column_dtype = "datetime"

    satisfied_formats = series.apply(
        get_satisfied_formats, unique_formats=unique_formats
    )
    unique_satisfied_formats_sorted = pd.Series(
        sum(satisfied_formats.values.tolist(), [])
    ).value_counts()
    satisfied_formats_sorted = satisfied_formats.apply(
        lambda x: sorted(x, key=unique_satisfied_formats_sorted.get, reverse=True)
    )
    if satisfied_formats_sorted.astype(str).nunique() == 1:
        parsed_dt = parse_by_format_with_valid_frequency(
            series, satisfied_formats_sorted.values[0]
        )
        column_dtype = "datetime"
    else:
        parsed_dt = pd.Series(
            [
                parse_datetime(series[i], satisfied_formats_sorted[i])
                for i in range(len(series))
            ]
        )

    if parsed_rate_check(parsed_dt):
        return series, "others"

    return parsed_dt, column_dtype
