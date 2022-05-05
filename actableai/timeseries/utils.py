import json
import re

from typing import Optional, Tuple, Union, Dict, Callable, Iterable, List, Any, Iterator
from copy import deepcopy

import numpy as np
import visions as v
import pandas as pd

from gluonts.model.forecast import Forecast


def find_gluonts_freq(freq: str) -> str:
    """Convert pandas frequency to GluonTS frequency.

    Args:
        freq: pandas frequency

    Returns:
        GluonTS frequency
    """
    # For W:
    if re.findall("\d*W", freq):
        return re.findall("\d*W", freq)[0]
    elif re.findall("\d*M", freq):
        return re.findall("\d*M", freq)[0]

    return freq


def interpolate(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Interpolate missing values in time series.

    Args:
        df: Time series DataFrame.
        freq: Frequency to use when interpolating

    Returns:
        Interpolated new DataFrame
    """
    return df.resample(freq).interpolate(method="linear")


def find_freq(pd_date: pd.Series, period: int = 10) -> Optional[str]:
    """Find the frequency from a list of datetime.

    Args:
        pd_date: List of datetime as a pandas Series, needs to be sorted.
        period: Window to look for when computing the frequency.

    Returns:
        The frequency or None if not found.
    """
    if len(pd_date) < 3:
        return None

    pd_date = pd_date.sort_values()
    freq = pd.infer_freq(pd_date)

    if freq:
        return freq

    infer_list = {}
    data_len = len(pd_date)
    for i in range(0, data_len - period, period):
        freq = pd.infer_freq(pd_date[i : i + period])
        if freq is not None:
            if freq not in infer_list:
                infer_list[str(freq)] = 0
            infer_list[str(freq)] += 1
    if len(infer_list) == 0:
        return None
    most_freq = max(infer_list, key=lambda freq: infer_list[freq])
    return most_freq


def get_satisfied_formats(row: pd.Series, unique_formats: List[str]) -> List[str]:
    """Find the datetime formats compatible with a list of datetime.

    Args:
        row: List of datetime as a pandas Series.
        unique_formats: List of formats to try.

    Returns:
        List of compatible formats.
    """
    satisfied_formats = []
    for dt_format in unique_formats:
        try:
            pd.to_datetime(row, format=dt_format)
            satisfied_formats.append(dt_format)
        except:
            continue

    return satisfied_formats


def parse_datetime(dt_str: pd.Series, formats: List[str]) -> Optional[pd.Series]:
    """Try to parse datetime using a list of formats. Returns the first working format.

    Args:
        dt_str: List of datetime as pandas Series.
        formats: List of datetime formats to try.

    Returns:
        Parsed list of datetime or None if no formats are compatible.
    """
    for fm in formats:
        try:
            result = pd.to_datetime(dt_str, format=fm)
            return result
        except Exception:
            pass
    return None


def parse_by_format_with_valid_frequency(
    series: pd.Series, formats: List[str]
) -> pd.Series:
    """Parse datetime using a list of formats. Returns the first working format.

    Args:
        series: List of datetime as pandas Series.
        formats: List of datetime formats to try.

    Returns:
        Parsed list of datetime.
    """
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
    """Parse datetime from a list of datetime.

    Args:
        series: List of datetime to parse as a pandas Series.
        min_parsed_rate: Ratio of minimum parsed dates.

    Returns:
        - Parsed pandas series.
        - Data type of the parsed series ["datetime", "others"].
    """
    from pandas._libs.tslibs.parsing import _guess_datetime_format

    if pd.api.types.is_datetime64_ns_dtype(series):
        return series, "datetime"

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


def dataframe_to_list_dataset(
    df_dict: Dict[Tuple[Any, ...], pd.DataFrame],
    target_columns: List[str],
    freq: str,
    *,
    real_static_feature_dict: Optional[Dict[Tuple[Any], List[float]]] = None,
    cat_static_feature_dict: Optional[Dict[Tuple[Any], List[Any]]] = None,
    real_dynamic_feature_columns: Optional[List[str]] = None,
    cat_dynamic_feature_columns: Optional[List[str]] = None,
    group_dict: Optional[Dict[Tuple[Any], int]] = None,
    prediction_length: Optional[int] = None,
    slice_df: Optional[Union[slice, Callable]] = None,
    training: bool = True,
) -> Iterable[Dict[str, Any]]:
    """Transform pandas DataFrame to GluonTS ListDataset.

    Args:
        df_dict: Dictionary containing the time series for each group.
        target_columns: List of columns to forecast.
        freq: Frequency of the time series.
        real_static_feature_dict: Dictionary containing a list of real features for
            each group.
        cat_static_feature_dict: Dictionary containing a list of categorical
            features for each group.
        real_dynamic_feature_columns: List of columns containing real features.
        cat_dynamic_feature_columns: List of columns containing categorical
            features.
        group_dict: Dictionary containing the unique label for each group.
        prediction_length: Length of the prediction to forecast. Cannot be None if
            `training` is False.
        slice_df: Slice or function to call that will return a slice. The slice will be
            applied to each group separately.
        training: If True the future dynamic features are trimmed out of the result.

    Returns:
        The ListDataset containing all the groups, features, and targets.
    """
    from gluonts.dataset.common import ListDataset

    if real_static_feature_dict is None:
        real_static_feature_dict = {}
    if cat_static_feature_dict is None:
        cat_static_feature_dict = {}
    if real_dynamic_feature_columns is None:
        real_dynamic_feature_columns = []
    if cat_dynamic_feature_columns is None:
        cat_dynamic_feature_columns = []
    if group_dict is None:
        group_dict = {}

    if prediction_length is None and not training:
        raise Exception("prediction length cannot be None if training is False")

    one_dim_target = len(target_columns) == 1
    if one_dim_target:
        target_columns = target_columns[0]

    slice_df = slice_df or slice(None)

    list_data = []
    for group in df_dict.keys():
        df = df_dict[group]

        slice_ = slice_df
        if callable(slice_df):
            slice_ = slice_df(df)

        df = df.iloc[slice_]

        entry = {"start": df.index[0]}

        real_static_features = real_static_feature_dict.get(group, [])
        cat_static_features = cat_static_feature_dict.get(group, [])

        if len(group_dict) > 0:
            cat_static_features = [group_dict[group], *cat_static_features]

        if len(real_static_features) > 0:
            entry["feat_static_real"] = real_static_features
        if len(cat_static_features) > 0:
            entry["feat_static_cat"] = cat_static_features
        if len(real_dynamic_feature_columns) > 0:
            entry["feat_dynamic_real"] = df[real_dynamic_feature_columns].to_numpy().T
        if len(cat_dynamic_feature_columns) > 0:
            entry["feat_dynamic_cat"] = df[cat_dynamic_feature_columns].to_numpy().T

        target = df[target_columns]
        if (
            len(real_dynamic_feature_columns) + len(cat_dynamic_feature_columns) > 0
            and not training
        ):
            target = target.iloc[:-prediction_length]
        entry["target"] = target.to_numpy().T

        list_data.append(entry)

    return ListDataset(list_data, freq, one_dim_target=one_dim_target)


def handle_features_dataset(
    dataset: Iterable[Dict[str, Any]],
    keep_feat_static_real: bool,
    keep_feat_static_cat: bool,
    keep_feat_dynamic_real: bool,
    keep_feat_dynamic_cat: bool,
) -> Iterable[Dict[str, Any]]:
    """Filter out features from GluonTS ListDataset.

    Args:
        dataset: The ListDataset to filter.
        keep_feat_static_real: Whether to keep the real static features or not.
        keep_feat_static_cat: Whether to keep the categorical static features or not.
        keep_feat_dynamic_real: Whether to keep to the real dynamic features or not.
        keep_feat_dynamic_cat: Whether to keep the categorical dynamic features or not.

    Returns:
        Filtered ListDataset.
    """
    from gluonts.transform import TransformedDataset
    from gluonts.dataset.common import ListDataset

    new_dataset = deepcopy(dataset)

    if isinstance(dataset, TransformedDataset):
        list_data = new_dataset.base_dataset.list_data
    elif isinstance(dataset, ListDataset):
        list_data = new_dataset.list_data
    else:
        raise Exception("Invalid dataset type")

    fields_to_exclude = []
    if not keep_feat_static_real:
        fields_to_exclude.append("feat_static_real")
    if not keep_feat_static_cat:
        fields_to_exclude.append("feat_static_cat")
    if not keep_feat_dynamic_real:
        fields_to_exclude.append("feat_dynamic_real")
    if not keep_feat_dynamic_cat:
        fields_to_exclude.append("feat_dynamic_cat")

    list_data = [
        {key: val for key, val in data.items() if key not in fields_to_exclude}
        for data in list_data
    ]

    if isinstance(dataset, TransformedDataset):
        new_dataset.base_dataset.list_data = list_data
    elif isinstance(dataset, ListDataset):
        new_dataset.list_data = list_data

    return new_dataset


def forecast_to_dataframe(
    forecast: Iterator[Forecast],
    target_columns: List[str],
    date_list: List[pd.datetime],
    quantiles: List[float] = [0.05, 0.5, 0.95],
) -> pd.DataFrame:
    """Convert GluonTS forecast to pandas DataFrame.

    Args:
        forecast: GluonTS forecast.
        target_columns: List of columns to forecast.
        date_list: List of datetime forecasted.
        quantiles: List of quantiles to forecast.

    Returns:
        Forecasted values as pandas DataFrame.
    """
    prediction_length = forecast.prediction_length

    quantiles_values_dict = {
        quantile: forecast.quantile(quantile).astype(float) for quantile in quantiles
    }

    if len(target_columns) <= 1:
        for quantile in quantiles_values_dict.keys():
            quantiles_values_dict[quantile] = quantiles_values_dict[quantile].reshape(
                prediction_length, 1
            )

    return pd.concat(
        [
            pd.DataFrame(
                {
                    "target": [target_column] * prediction_length,
                    "date": date_list,
                    **{
                        str(quantile): quantiles_values[:, index]
                        for quantile, quantiles_values in quantiles_values_dict.items()
                    },
                }
            )
            for index, target_column in enumerate(target_columns)
        ],
        ignore_index=True,
    )
