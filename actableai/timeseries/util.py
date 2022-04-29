from copy import deepcopy
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


def find_gluonts_freq(freq):
    # For W:
    if re.findall("\d*W", freq):
        return re.findall("\d*W", freq)[0]
    elif re.findall("\d*M", freq):
        return re.findall("\d*M", freq)[0]

    return freq


def interpolate(df, freq):
    return df.resample(freq).interpolate(method="linear")


def find_freq(
    pd_date: pd.Series, period=10
) -> Optional[str]:  ## Need to sorted before.
    if len(pd_date) < 3:
        return None
    pd_date = pd_date.sort_values()  # Sorting them ?
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


# FIXME unused
def minmax_scaler_fit_transform(df):
    from sklearn.preprocessing import MinMaxScaler

    x = df.values
    scaler = MinMaxScaler()
    x_scaled = scaler.fit_transform(x)
    x_scaled = pd.DataFrame(x_scaled)
    x_scaled.index = df.index

    return x_scaled, scaler


# FIXME unused
def minmax_scaler_transform(df, scaler):

    x = df.values
    x_scaled = scaler.transform(x)
    x_scaled = pd.DataFrame(x_scaled)
    x_scaled.index = df.index

    return x_scaled


# FIXME unused
def inverse_transform(normalized_fc, scaler):
    fc_samples = normalized_fc.samples
    fc_shape = fc_samples.shape
    fc_samples = scaler.inverse_transform(fc_samples.reshape(-1, fc_shape[2]))
    normalized_fc.samples = fc_samples.reshape(fc_shape)

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
    """
    TODO write documentation
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
    df_dict,
    target_columns,
    freq,
    *,
    real_static_feature_dict=None,
    cat_static_feature_dict=None,
    real_dynamic_feature_columns=None,
    cat_dynamic_feature_columns=None,
    group_dict=None,
    prediction_length=None,
    slice_df=None,
    training=True,
):
    """
    TODO write documentation
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
        raise Exception("prediction length cannot be None if trainig is False")

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
    dataset,
    keep_feat_static_real,
    keep_feat_static_cat,
    keep_feat_dynamic_real,
    keep_feat_dynamic_cat,
):
    """
    TODO write documentation
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


def forecast_to_dataframe(forecast, target_columns, date_list):
    prediction_length = forecast.prediction_length

    q5_quantile = forecast.quantile(0.05).astype(float)
    q50_quantile = forecast.quantile(0.5).astype(float)
    q95_quantile = forecast.quantile(0.95).astype(float)

    if len(target_columns) <= 1:
        q5_quantile = q5_quantile.reshape(prediction_length, 1)
        q50_quantile = q50_quantile.reshape(prediction_length, 1)
        q95_quantile = q95_quantile.reshape(prediction_length, 1)

    return pd.concat(
        [
            pd.DataFrame(
                {
                    "target": [target_column] * prediction_length,
                    "date": date_list,
                    "q5": q5_quantile[:, index],
                    "q50": q50_quantile[:, index],
                    "q95": q95_quantile[:, index],
                }
            )
            for index, target_column in enumerate(target_columns)
        ],
        ignore_index=True,
    )


def generate_train_valid_data(
    df_dict,
    target_columns,
    prediction_length,
    freq,
    group_dict,
    real_static_feature_dict,
    cat_static_feature_dict,
    real_dynamic_feature_columns,
    cat_dynamic_feature_columns,
    tune_samples,
    sampling_method="random",
):
    from gluonts.dataset.common import ListDataset
    """
    TODO write documentation
    """
    train_data = dataframe_to_list_dataset(
        df_dict,
        target_columns,
        freq,
        real_static_feature_dict=real_static_feature_dict,
        cat_static_feature_dict=cat_static_feature_dict,
        real_dynamic_feature_columns=real_dynamic_feature_columns,
        cat_dynamic_feature_columns=cat_dynamic_feature_columns,
        group_dict=group_dict,
        prediction_length=prediction_length,
        training=True,
    )

    train_data_partial = dataframe_to_list_dataset(
        df_dict,
        target_columns,
        freq,
        real_static_feature_dict=real_static_feature_dict,
        cat_static_feature_dict=cat_static_feature_dict,
        real_dynamic_feature_columns=real_dynamic_feature_columns,
        cat_dynamic_feature_columns=cat_dynamic_feature_columns,
        group_dict=group_dict,
        prediction_length=prediction_length,
        slice_df=slice(-prediction_length - tune_samples),
        training=True,
    )

    tune_data_list = []
    for i in range(tune_samples):
        slice_function = None
        if sampling_method == "random":
            slice_function = lambda df: slice(
                np.random.randint(2 * prediction_length + 1, df.shape[0] + 1)
            )
        elif sampling_method == "last":
            slice_function = lambda df: slice(df.shape[0] - i)
        else:
            raise Exception("Unkown sampling method")

        tune_data_list.append(
            dataframe_to_list_dataset(
                df_dict,
                target_columns,
                freq,
                real_static_feature_dict=real_static_feature_dict,
                cat_static_feature_dict=cat_static_feature_dict,
                real_dynamic_feature_columns=real_dynamic_feature_columns,
                cat_dynamic_feature_columns=cat_dynamic_feature_columns,
                group_dict=group_dict,
                prediction_length=prediction_length,
                slice_df=slice_function,
                training=True,
            )
        )

    # Merge all samples into the same ListDataset
    tune_data = None
    if len(tune_data_list) > 0:
        list_data = []

        for tune_data_sample in tune_data_list:
            list_data += tune_data_sample.list_data

        tune_data = ListDataset(
            list_data, freq, one_dim_target=(len(target_columns) == 1)
        )

    return train_data, train_data_partial, tune_data
