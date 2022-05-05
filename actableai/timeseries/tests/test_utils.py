import pytest
import datetime
from copy import copy

import pandas as pd
import numpy as np

from gluonts.dataset.common import ListDataset
from gluonts.transform import TransformedDataset

from actableai.timeseries.utils import (
    find_freq,
    dataframe_to_list_dataset,
    find_gluonts_freq,
    handle_features_dataset,
    handle_datetime_column,
)
from actableai.utils.testing import generate_forecast_df_dict, generate_date_range


# Test find_freq
@pytest.mark.parametrize("freq", ["H", "S", "T", "N"])
@pytest.mark.parametrize("freq_n", ["", "15", "2", "30"])
def test_find_freq(np_rng, freq, freq_n):
    freq = freq_n + freq
    pd_date = pd.Series(generate_date_range(np_rng, freq=freq))
    assert find_freq(pd_date) == freq


@pytest.mark.parametrize("freq", ["H", "S", "T", "N"])
@pytest.mark.parametrize("freq_n", ["", "15", "2", "30"])
def test_find_freq_missing_values(np_rng, freq, freq_n):
    freq = freq_n + freq
    pd_date = pd.Series(generate_date_range(np_rng, periods=10, freq=freq))
    # Remove two values in the ten values
    pd_date[3] = np.nan
    pd_date[7] = np.nan
    assert find_freq(pd_date, period=3) == freq


def test_find_freq_not_enough_values(np_rng):
    pd_date = pd.Series(generate_date_range(np_rng, periods=2))
    assert find_freq(pd_date) is None


def test_find_freq_non_sense():
    assert find_freq(pd.Series(["01/02/2012", "03/03/2037", "01/01/1997"])) is None


@pytest.mark.parametrize(
    "start_date",
    ["2021-02-06", "2015-09-08", "18:00", "18:30:25", "2020-01-02 05:00:00+02:00"],
)
@pytest.mark.parametrize("freq", ["T", "H", "S", "Y", "us"])
def test_handle_datetime_column_pd_datetime(start_date, freq):
    date_range = pd.Series(pd.date_range(start_date, periods=45, freq=freq))
    parsed_dt, dtype = handle_datetime_column(date_range)

    assert parsed_dt is not None
    assert dtype == "datetime"
    assert (parsed_dt == date_range).all()


@pytest.mark.parametrize(
    "start_date", ["2021-02-06", "2015-09-08", "2020-01-02 05:00:00+02:00"]
)
@pytest.mark.parametrize("freq", ["T", "H", "S", "Y", "us"])
def test_handle_datetime_column_str(start_date, freq):
    date_range = pd.Series(pd.date_range(start_date, periods=10, freq=freq))
    parsed_dt, dtype = handle_datetime_column(date_range.astype(str))

    assert parsed_dt is not None
    assert dtype == "datetime"
    for i in range(len(date_range)):
        revert_date = None
        if date_range[i].day <= 12:
            revert_date = date_range[i].replace(
                month=date_range[i].day, day=date_range[i].month
            )
        assert parsed_dt[i] == date_range[i] or parsed_dt[i] == revert_date


def test_handle_datetime_column_mixed_hour():
    date_range = pd.Series(pd.date_range("18:00", periods=10, freq="H"))
    parsed_dt, dtype = handle_datetime_column(date_range.astype(str))

    assert parsed_dt is not None
    assert dtype == "datetime"
    assert (parsed_dt.dt.time == date_range.dt.time).all()


@pytest.mark.parametrize("freq", ["T", "S", "Y", "us"])
def test_handle_datetime_column_hour_multi_freq(freq):
    date_range = pd.Series(pd.date_range("18:00", periods=10, freq=freq))
    parsed_dt, dtype = handle_datetime_column(date_range.astype(str))

    assert parsed_dt is not None
    assert dtype == "datetime"
    assert (parsed_dt.dt.time == date_range.dt.time).all()


def test_handle_datetime_column_tstamp_mixed():
    date_range = pd.Series(pd.date_range("18:30:25", periods=10, freq="H"))
    parsed_dt, dtype = handle_datetime_column(date_range.astype(str))

    assert parsed_dt is not None
    assert dtype == "datetime"
    assert (parsed_dt.dt.time == date_range.dt.time).all()


@pytest.mark.parametrize("freq", ["T", "S", "Y", "us"])
def test_handle_datetime_column_tstamp_multi_freq(freq):
    date_range = pd.Series(pd.date_range("18:30:25", periods=10, freq=freq))
    parsed_dt, dtype = handle_datetime_column(date_range.astype(str))

    assert parsed_dt is not None
    assert dtype == "datetime"
    assert (parsed_dt.dt.time == date_range.dt.time).all()


def test_handle_datetime_column_dotted():
    date_range = pd.Series(["{}.{}.20{:0>2}".format(i, i, i) for i in range(1, 10)])
    parsed_dt, dtype = handle_datetime_column(date_range)

    assert parsed_dt is not None
    assert dtype == "datetime"
    for i in range(len(date_range)):
        assert parsed_dt[i] == pd.Timestamp(date_range[i])


def test_handle_datetime_column_nan_values_threshold():
    date_range = pd.Series(["{}.{}.20{:0>2}".format(i, i, i) for i in range(1, 10)])
    date_range[0:3] = ["abc" for _ in range(3)]
    parsed_dt, dtype = handle_datetime_column(date_range)

    assert parsed_dt is not None
    assert dtype == "datetime"
    assert parsed_dt.isna().sum() == 3
    for i in range(len(date_range)):
        if i >= 0 and i < 3:
            assert pd.isnull(parsed_dt[i])
        else:
            assert parsed_dt[i] == pd.Timestamp(date_range[i])


def test_handle_datetime_column_nan_values_no_threshold():
    date_range = pd.Series(["{}.{}.20{:0>2}".format(i, i, i) for i in range(1, 10)])
    date_range[0:7] = ["abc" for _ in range(7)]
    parsed_dt, dtype = handle_datetime_column(date_range)

    assert parsed_dt is not None
    assert dtype == "others"
    assert (parsed_dt == date_range).all()


# Test dataframe_to_list_dataset
@pytest.mark.parametrize("n_groups", [1, 5])
@pytest.mark.parametrize("n_targets", [1, 5])
@pytest.mark.parametrize("freq", ["T"])
def test_dataframe_to_list_dataset_simple(np_rng, n_groups, n_targets, freq):
    df_dict, target_columns, _, _ = generate_forecast_df_dict(
        np_rng, n_groups, n_targets=n_targets, freq=freq
    )
    gluonts_freq = find_gluonts_freq(freq)

    list_dataset = dataframe_to_list_dataset(
        df_dict,
        target_columns,
        gluonts_freq,
    )

    assert list_dataset is not None
    assert isinstance(list_dataset, ListDataset)
    assert len(list_dataset.list_data) == len(df_dict)

    for group_index, (group_name, df_group) in enumerate(df_dict.items()):
        data = list_dataset.list_data[group_index]

        assert data["start"] == df_group.index[0]

        if n_targets > 1:
            for target_index, target in enumerate(target_columns):
                assert (data["target"][target_index, :] == df_group[target]).all()
        else:
            assert (data["target"] == df_group[target_columns[0]]).all()

        assert "feat_static_real" not in data
        assert "feat_static_cat" not in data
        assert "feat_dynamic_real" not in data
        assert "feat_dynamic_cat" not in data


@pytest.mark.parametrize("n_groups", [1, 5])
@pytest.mark.parametrize("n_targets", [1, 5])
@pytest.mark.parametrize("freq", ["T"])
def test_dataframe_to_list_dataset_real_static_features(
    np_rng, n_groups, n_targets, freq
):
    df_dict, target_columns, _, _ = generate_forecast_df_dict(
        np_rng, n_groups, n_targets=n_targets, freq=freq
    )
    gluonts_freq = find_gluonts_freq(freq)

    real_static_feature_dict = {}
    for group in df_dict.keys():
        n_features = np_rng.integers(2, 10)
        real_static_feature_dict[group] = np_rng.standard_normal(n_features)

    list_dataset = dataframe_to_list_dataset(
        df_dict,
        target_columns,
        gluonts_freq,
        real_static_feature_dict=real_static_feature_dict,
    )

    assert list_dataset is not None
    assert isinstance(list_dataset, ListDataset)
    assert len(list_dataset.list_data) == len(df_dict)

    for group_index, (group_name, df_group) in enumerate(df_dict.items()):
        data = list_dataset.list_data[group_index]

        assert data["start"] == df_group.index[0]

        if n_targets > 1:
            for target_index, target in enumerate(target_columns):
                assert (data["target"][target_index, :] == df_group[target]).all()
        else:
            assert (data["target"] == df_group[target_columns[0]]).all()

        for feat in real_static_feature_dict[group_name]:
            assert feat in data["feat_static_real"]
        assert "feat_static_cat" not in data
        assert "feat_dynamic_real" not in data
        assert "feat_dynamic_cat" not in data


@pytest.mark.parametrize("n_groups", [1, 5])
@pytest.mark.parametrize("n_targets", [1, 5])
@pytest.mark.parametrize("freq", ["T"])
def test_dataframe_to_list_dataset_cat_static_features(
    np_rng, n_groups, n_targets, freq
):
    df_dict, target_columns, _, _ = generate_forecast_df_dict(
        np_rng, n_groups, n_targets=n_targets, freq=freq
    )
    gluonts_freq = find_gluonts_freq(freq)

    cat_static_feature_dict = {}
    for group in df_dict.keys():
        n_features = np_rng.integers(2, 10)
        cat_static_feature_dict[group] = np_rng.integers(1, 10, n_features)

    list_dataset = dataframe_to_list_dataset(
        df_dict,
        target_columns,
        gluonts_freq,
        cat_static_feature_dict=cat_static_feature_dict,
    )

    assert list_dataset is not None
    assert isinstance(list_dataset, ListDataset)
    assert len(list_dataset.list_data) == len(df_dict)

    for group_index, (group_name, df_group) in enumerate(df_dict.items()):
        data = list_dataset.list_data[group_index]

        assert data["start"] == df_group.index[0]

        if n_targets > 1:
            for target_index, target in enumerate(target_columns):
                assert (data["target"][target_index, :] == df_group[target]).all()
        else:
            assert (data["target"] == df_group[target_columns[0]]).all()

        for feat in cat_static_feature_dict[group_name]:
            assert feat in data["feat_static_cat"]
        assert "feat_static_real" not in data
        assert "feat_dynamic_real" not in data
        assert "feat_dynamic_cat" not in data


@pytest.mark.parametrize("n_groups", [1, 5])
@pytest.mark.parametrize("n_targets", [1, 5])
@pytest.mark.parametrize("freq", ["T"])
def test_dataframe_to_list_dataset_real_dynamic_features(
    np_rng, n_groups, n_targets, freq
):
    (
        df_dict,
        target_columns,
        real_dynamic_feature_columns,
        _,
    ) = generate_forecast_df_dict(
        np_rng,
        n_groups,
        n_targets=n_targets,
        freq=freq,
        n_real_features=np_rng.integers(1, 10),
    )
    gluonts_freq = find_gluonts_freq(freq)

    list_dataset = dataframe_to_list_dataset(
        df_dict,
        target_columns,
        gluonts_freq,
        real_dynamic_feature_columns=real_dynamic_feature_columns,
    )

    assert list_dataset is not None
    assert isinstance(list_dataset, ListDataset)
    assert len(list_dataset.list_data) == len(df_dict)

    for group_index, (group_name, df_group) in enumerate(df_dict.items()):
        data = list_dataset.list_data[group_index]

        assert data["start"] == df_group.index[0]

        if n_targets > 1:
            for target_index, target in enumerate(target_columns):
                assert (data["target"][target_index, :] == df_group[target]).all()
        else:
            assert (data["target"] == df_group[target_columns[0]]).all()

        for feat_index, feat_column in enumerate(real_dynamic_feature_columns):
            assert (
                data["feat_dynamic_real"][feat_index, :] == df_group[feat_column]
            ).all()
        assert "feat_static_real" not in data
        assert "feat_static_cat" not in data
        assert "feat_dynamic_cat" not in data


@pytest.mark.parametrize("n_groups", [1, 5])
@pytest.mark.parametrize("n_targets", [1, 5])
@pytest.mark.parametrize("freq", ["T"])
def test_dataframe_to_list_dataset_cat_dynamic_features(
    np_rng, n_groups, n_targets, freq
):
    df_dict, target_columns, _, cat_dynamic_feature_columns = generate_forecast_df_dict(
        np_rng,
        n_groups,
        n_targets=n_targets,
        freq=freq,
        n_cat_features=np_rng.integers(1, 10),
    )
    gluonts_freq = find_gluonts_freq(freq)

    list_dataset = dataframe_to_list_dataset(
        df_dict,
        target_columns,
        gluonts_freq,
        cat_dynamic_feature_columns=cat_dynamic_feature_columns,
    )

    assert list_dataset is not None
    assert isinstance(list_dataset, ListDataset)
    assert len(list_dataset.list_data) == len(df_dict)

    for group_index, (group_name, df_group) in enumerate(df_dict.items()):
        data = list_dataset.list_data[group_index]

        assert data["start"] == df_group.index[0]

        if n_targets > 1:
            for target_index, target in enumerate(target_columns):
                assert (data["target"][target_index, :] == df_group[target]).all()
        else:
            assert (data["target"] == df_group[target_columns[0]]).all()

        for feat_index, feat_column in enumerate(cat_dynamic_feature_columns):
            assert (
                data["feat_dynamic_cat"][feat_index, :] == df_group[feat_column]
            ).all()
        assert "feat_static_real" not in data
        assert "feat_static_cat" not in data
        assert "feat_dynamic_real" not in data


@pytest.mark.parametrize("n_groups", [1, 5])
@pytest.mark.parametrize("n_targets", [1, 5])
@pytest.mark.parametrize("freq", ["T"])
def test_dataframe_to_list_dataset_group_dict(np_rng, n_groups, n_targets, freq):
    df_dict, target_columns, _, _ = generate_forecast_df_dict(
        np_rng, n_groups, n_targets=n_targets, freq=freq
    )
    gluonts_freq = find_gluonts_freq(freq)

    group_dict = None
    if n_groups > 1:
        group_dict = {
            group: group_index for group_index, group in enumerate(df_dict.keys())
        }

    list_dataset = dataframe_to_list_dataset(
        df_dict, target_columns, gluonts_freq, group_dict=group_dict
    )

    assert list_dataset is not None
    assert isinstance(list_dataset, ListDataset)
    assert len(list_dataset.list_data) == len(df_dict)

    for group_index, (group_name, df_group) in enumerate(df_dict.items()):
        data = list_dataset.list_data[group_index]

        assert data["start"] == df_group.index[0]

        if n_targets > 1:
            for target_index, target in enumerate(target_columns):
                assert (data["target"][target_index, :] == df_group[target]).all()
        else:
            assert (data["target"] == df_group[target_columns[0]]).all()

        if n_groups > 1:
            assert group_dict[group_name] in data["feat_static_cat"]
        else:
            assert "feat_static_cat" not in data
        assert "feat_static_real" not in data
        assert "feat_dynamic_real" not in data
        assert "feat_dynamic_cat" not in data


@pytest.mark.parametrize("n_groups", [1, 5])
@pytest.mark.parametrize("n_targets", [1, 5])
@pytest.mark.parametrize("freq", ["T"])
def test_dataframe_to_list_dataset_no_training(np_rng, n_groups, n_targets, freq):
    (
        df_dict,
        target_columns,
        real_dynamic_feature_columns,
        cat_dynamic_feature_columns,
    ) = generate_forecast_df_dict(
        np_rng,
        n_groups,
        n_targets=n_targets,
        freq=freq,
        n_real_features=np_rng.integers(1, 10),
        n_cat_features=np_rng.integers(1, 10),
    )
    gluonts_freq = find_gluonts_freq(freq)

    prediction_length = np_rng.integers(1, 5)

    list_dataset = dataframe_to_list_dataset(
        df_dict,
        target_columns,
        gluonts_freq,
        real_dynamic_feature_columns=real_dynamic_feature_columns,
        cat_dynamic_feature_columns=cat_dynamic_feature_columns,
        prediction_length=prediction_length,
        training=False,
    )

    assert list_dataset is not None
    assert isinstance(list_dataset, ListDataset)
    assert len(list_dataset.list_data) == len(df_dict)

    for group_index, (group_name, df_group) in enumerate(df_dict.items()):
        data = list_dataset.list_data[group_index]

        assert data["start"] == df_group.index[0]

        if n_targets > 1:
            for target_index, target in enumerate(target_columns):
                assert (
                    data["target"][target_index, :]
                    == df_group[target][:-prediction_length]
                ).all()
        else:
            assert (
                data["target"] == df_group[target_columns[0]][:-prediction_length]
            ).all()

        for feat_index, feat_column in enumerate(real_dynamic_feature_columns):
            assert (
                data["feat_dynamic_real"][feat_index, :] == df_group[feat_column]
            ).all()
        for feat_index, feat_column in enumerate(cat_dynamic_feature_columns):
            assert (
                data["feat_dynamic_cat"][feat_index, :] == df_group[feat_column]
            ).all()
        assert "feat_static_real" not in data
        assert "feat_static_cat" not in data


@pytest.mark.parametrize("n_groups", [1, 5])
@pytest.mark.parametrize("n_targets", [1, 5])
@pytest.mark.parametrize("freq", ["T"])
def test_dataframe_to_list_dataset_static_slice(np_rng, n_groups, n_targets, freq):
    (
        df_dict,
        target_columns,
        real_dynamic_feature_columns,
        cat_dynamic_feature_columns,
    ) = generate_forecast_df_dict(
        np_rng,
        n_groups,
        n_targets=n_targets,
        freq=freq,
        n_real_features=np_rng.integers(1, 10),
        n_cat_features=np_rng.integers(1, 10),
    )
    gluonts_freq = find_gluonts_freq(freq)

    slice_df = slice(0, np_rng.integers(2, 5))

    list_dataset = dataframe_to_list_dataset(
        df_dict,
        target_columns,
        gluonts_freq,
        real_dynamic_feature_columns=real_dynamic_feature_columns,
        cat_dynamic_feature_columns=cat_dynamic_feature_columns,
        slice_df=slice_df,
    )

    assert list_dataset is not None
    assert isinstance(list_dataset, ListDataset)
    assert len(list_dataset.list_data) == len(df_dict)

    for group_index, (group_name, df_group) in enumerate(df_dict.items()):
        data = list_dataset.list_data[group_index]

        assert data["start"] == df_group.index[0]

        if n_targets > 1:
            for target_index, target in enumerate(target_columns):
                assert (
                    data["target"][target_index, :] == df_group[target][slice_df]
                ).all()
        else:
            assert (data["target"] == df_group[target_columns[0]][slice_df]).all()

        for feat_index, feat_column in enumerate(real_dynamic_feature_columns):
            assert (
                data["feat_dynamic_real"][feat_index, :]
                == df_group[feat_column][slice_df]
            ).all()
        for feat_index, feat_column in enumerate(cat_dynamic_feature_columns):
            assert (
                data["feat_dynamic_cat"][feat_index, :]
                == df_group[feat_column][slice_df]
            ).all()
        assert "feat_static_real" not in data
        assert "feat_static_cat" not in data


@pytest.mark.parametrize("n_groups", [1, 5])
@pytest.mark.parametrize("n_targets", [1, 5])
@pytest.mark.parametrize("freq", ["T"])
def test_dataframe_to_list_dataset_dynamic_slice(np_rng, n_groups, n_targets, freq):
    (
        df_dict,
        target_columns,
        real_dynamic_feature_columns,
        cat_dynamic_feature_columns,
    ) = generate_forecast_df_dict(
        np_rng,
        n_groups,
        n_targets=n_targets,
        freq=freq,
        n_real_features=np_rng.integers(1, 10),
        n_cat_features=np_rng.integers(1, 10),
    )
    gluonts_freq = find_gluonts_freq(freq)

    slice_df = lambda df: slice(0, int(len(df) * 0.8))

    list_dataset = dataframe_to_list_dataset(
        df_dict,
        target_columns,
        gluonts_freq,
        real_dynamic_feature_columns=real_dynamic_feature_columns,
        cat_dynamic_feature_columns=cat_dynamic_feature_columns,
        slice_df=slice_df,
    )

    assert list_dataset is not None
    assert isinstance(list_dataset, ListDataset)
    assert len(list_dataset.list_data) == len(df_dict)

    for group_index, (group_name, df_group) in enumerate(df_dict.items()):
        slice_ = slice_df(df_group)
        data = list_dataset.list_data[group_index]

        assert data["start"] == df_group.index[0]

        if n_targets > 1:
            for target_index, target in enumerate(target_columns):
                assert (
                    data["target"][target_index, :] == df_group[target][slice_]
                ).all()
        else:
            assert (data["target"] == df_group[target_columns[0]][slice_]).all()

        for feat_index, feat_column in enumerate(real_dynamic_feature_columns):
            assert (
                data["feat_dynamic_real"][feat_index, :]
                == df_group[feat_column][slice_]
            ).all()
        for feat_index, feat_column in enumerate(cat_dynamic_feature_columns):
            assert (
                data["feat_dynamic_cat"][feat_index, :] == df_group[feat_column][slice_]
            ).all()
        assert "feat_static_real" not in data
        assert "feat_static_cat" not in data


@pytest.mark.parametrize("dataset_type", ["list_dataset", "transformed_dataset"])
@pytest.mark.parametrize("n_groups", [1, 5])
@pytest.mark.parametrize("n_targets", [1, 5])
@pytest.mark.parametrize("freq", ["T"])
def test_handle_features_dataset_remove_static_real(
    dataset_type, np_rng, n_groups, n_targets, freq
):
    (
        df_dict,
        target_columns,
        real_dynamic_feature_columns,
        cat_dynamic_feature_columns,
    ) = generate_forecast_df_dict(
        np_rng,
        n_groups,
        n_targets=n_targets,
        freq=freq,
        n_real_features=np_rng.integers(1, 10),
        n_cat_features=np_rng.integers(1, 10),
    )
    gluonts_freq = find_gluonts_freq(freq)

    real_static_feature_dict = {}
    for group in df_dict.keys():
        n_features = np_rng.integers(2, 10)
        real_static_feature_dict[group] = np_rng.standard_normal(n_features)

    cat_static_feature_dict = {}
    for group in df_dict.keys():
        n_features = np_rng.integers(2, 10)
        cat_static_feature_dict[group] = np_rng.standard_normal(n_features)

    original_dataset = dataframe_to_list_dataset(
        df_dict,
        target_columns,
        gluonts_freq,
        real_static_feature_dict=real_static_feature_dict,
        cat_static_feature_dict=cat_static_feature_dict,
        real_dynamic_feature_columns=real_dynamic_feature_columns,
        cat_dynamic_feature_columns=cat_dynamic_feature_columns,
    )

    if dataset_type == "transformed_dataset":
        original_dataset = TransformedDataset(original_dataset, None)

    keep_feat_static_real = False
    keep_feat_static_cat = True
    keep_feat_dynamic_real = True
    keep_feat_dynamic_cat = True

    clean_dataset = handle_features_dataset(
        original_dataset,
        keep_feat_static_real,
        keep_feat_static_cat,
        keep_feat_dynamic_real,
        keep_feat_dynamic_cat,
    )

    if dataset_type == "transformed_dataset":
        original_list_data = original_dataset.base_dataset.list_data
        clean_list_data = clean_dataset.base_dataset.list_data
    else:
        original_list_data = original_dataset.list_data
        clean_list_data = clean_dataset.list_data

    assert len(clean_list_data) == len(df_dict)

    for group_index, (group_name, df_group) in enumerate(df_dict.items()):
        original_data = original_list_data[group_index]
        data = clean_list_data[group_index]

        assert data["start"] == df_group.index[0]

        if n_targets > 1:
            for target_index, target in enumerate(target_columns):
                assert (data["target"][target_index, :] == df_group[target]).all()
        else:
            assert (data["target"] == df_group[target_columns[0]]).all()

        assert "feat_static_real" not in data
        for feat in real_static_feature_dict[group_name]:
            assert feat in original_data["feat_static_real"]

        for feat in cat_static_feature_dict[group_name]:
            assert feat in data["feat_static_cat"]
        for feat_index, feat_column in enumerate(real_dynamic_feature_columns):
            assert (
                data["feat_dynamic_real"][feat_index, :] == df_group[feat_column]
            ).all()
        for feat_index, feat_column in enumerate(cat_dynamic_feature_columns):
            assert (
                data["feat_dynamic_cat"][feat_index, :] == df_group[feat_column]
            ).all()


@pytest.mark.parametrize("dataset_type", ["list_dataset", "transformed_dataset"])
@pytest.mark.parametrize("n_groups", [1, 5])
@pytest.mark.parametrize("n_targets", [1, 5])
@pytest.mark.parametrize("freq", ["T"])
def test_handle_features_dataset_remove_static_cat(
    dataset_type, np_rng, n_groups, n_targets, freq
):
    (
        df_dict,
        target_columns,
        real_dynamic_feature_columns,
        cat_dynamic_feature_columns,
    ) = generate_forecast_df_dict(
        np_rng,
        n_groups,
        n_targets=n_targets,
        freq=freq,
        n_real_features=np_rng.integers(1, 10),
        n_cat_features=np_rng.integers(1, 10),
    )
    gluonts_freq = find_gluonts_freq(freq)

    real_static_feature_dict = {}
    for group in df_dict.keys():
        n_features = np_rng.integers(2, 10)
        real_static_feature_dict[group] = np_rng.standard_normal(n_features)

    cat_static_feature_dict = {}
    for group in df_dict.keys():
        n_features = np_rng.integers(2, 10)
        cat_static_feature_dict[group] = np_rng.standard_normal(n_features)

    original_dataset = dataframe_to_list_dataset(
        df_dict,
        target_columns,
        gluonts_freq,
        real_static_feature_dict=real_static_feature_dict,
        cat_static_feature_dict=cat_static_feature_dict,
        real_dynamic_feature_columns=real_dynamic_feature_columns,
        cat_dynamic_feature_columns=cat_dynamic_feature_columns,
    )

    if dataset_type == "transformed_dataset":
        original_dataset = TransformedDataset(original_dataset, None)

    keep_feat_static_real = True
    keep_feat_static_cat = False
    keep_feat_dynamic_real = True
    keep_feat_dynamic_cat = True

    clean_dataset = handle_features_dataset(
        original_dataset,
        keep_feat_static_real,
        keep_feat_static_cat,
        keep_feat_dynamic_real,
        keep_feat_dynamic_cat,
    )

    if dataset_type == "transformed_dataset":
        original_list_data = original_dataset.base_dataset.list_data
        clean_list_data = clean_dataset.base_dataset.list_data
    else:
        original_list_data = original_dataset.list_data
        clean_list_data = clean_dataset.list_data

    assert len(clean_list_data) == len(df_dict)

    for group_index, (group_name, df_group) in enumerate(df_dict.items()):
        original_data = original_list_data[group_index]
        data = clean_list_data[group_index]

        assert data["start"] == df_group.index[0]

        if n_targets > 1:
            for target_index, target in enumerate(target_columns):
                assert (data["target"][target_index, :] == df_group[target]).all()
        else:
            assert (data["target"] == df_group[target_columns[0]]).all()

        assert "feat_static_cat" not in data
        for feat in cat_static_feature_dict[group_name]:
            assert feat in original_data["feat_static_cat"]

        for feat in real_static_feature_dict[group_name]:
            assert feat in data["feat_static_real"]
        for feat_index, feat_column in enumerate(real_dynamic_feature_columns):
            assert (
                data["feat_dynamic_real"][feat_index, :] == df_group[feat_column]
            ).all()
        for feat_index, feat_column in enumerate(cat_dynamic_feature_columns):
            assert (
                data["feat_dynamic_cat"][feat_index, :] == df_group[feat_column]
            ).all()


@pytest.mark.parametrize("dataset_type", ["list_dataset", "transformed_dataset"])
@pytest.mark.parametrize("n_groups", [1, 5])
@pytest.mark.parametrize("n_targets", [1, 5])
@pytest.mark.parametrize("freq", ["T"])
def test_handle_features_dataset_remove_dynamic_real(
    dataset_type, np_rng, n_groups, n_targets, freq
):
    (
        df_dict,
        target_columns,
        real_dynamic_feature_columns,
        cat_dynamic_feature_columns,
    ) = generate_forecast_df_dict(
        np_rng,
        n_groups,
        n_targets=n_targets,
        freq=freq,
        n_real_features=np_rng.integers(1, 10),
        n_cat_features=np_rng.integers(1, 10),
    )
    gluonts_freq = find_gluonts_freq(freq)

    real_static_feature_dict = {}
    for group in df_dict.keys():
        n_features = np_rng.integers(2, 10)
        real_static_feature_dict[group] = np_rng.standard_normal(n_features)

    cat_static_feature_dict = {}
    for group in df_dict.keys():
        n_features = np_rng.integers(2, 10)
        cat_static_feature_dict[group] = np_rng.standard_normal(n_features)

    original_dataset = dataframe_to_list_dataset(
        df_dict,
        target_columns,
        gluonts_freq,
        real_static_feature_dict=real_static_feature_dict,
        cat_static_feature_dict=cat_static_feature_dict,
        real_dynamic_feature_columns=real_dynamic_feature_columns,
        cat_dynamic_feature_columns=cat_dynamic_feature_columns,
    )

    if dataset_type == "transformed_dataset":
        original_dataset = TransformedDataset(original_dataset, None)

    keep_feat_static_real = True
    keep_feat_static_cat = True
    keep_feat_dynamic_real = False
    keep_feat_dynamic_cat = True

    clean_dataset = handle_features_dataset(
        original_dataset,
        keep_feat_static_real,
        keep_feat_static_cat,
        keep_feat_dynamic_real,
        keep_feat_dynamic_cat,
    )

    if dataset_type == "transformed_dataset":
        original_list_data = original_dataset.base_dataset.list_data
        clean_list_data = clean_dataset.base_dataset.list_data
    else:
        original_list_data = original_dataset.list_data
        clean_list_data = clean_dataset.list_data

    assert len(clean_list_data) == len(df_dict)

    for group_index, (group_name, df_group) in enumerate(df_dict.items()):
        original_data = original_list_data[group_index]
        data = clean_list_data[group_index]

        assert data["start"] == df_group.index[0]

        if n_targets > 1:
            for target_index, target in enumerate(target_columns):
                assert (data["target"][target_index, :] == df_group[target]).all()
        else:
            assert (data["target"] == df_group[target_columns[0]]).all()

        assert "feat_dynamic_real" not in data
        for feat_index, feat_column in enumerate(real_dynamic_feature_columns):
            assert (
                original_data["feat_dynamic_real"][feat_index, :]
                == df_group[feat_column]
            ).all()

        for feat in real_static_feature_dict[group_name]:
            assert feat in data["feat_static_real"]
        for feat in cat_static_feature_dict[group_name]:
            assert feat in data["feat_static_cat"]
        for feat_index, feat_column in enumerate(cat_dynamic_feature_columns):
            assert (
                data["feat_dynamic_cat"][feat_index, :] == df_group[feat_column]
            ).all()


@pytest.mark.parametrize("dataset_type", ["list_dataset", "transformed_dataset"])
@pytest.mark.parametrize("n_groups", [1, 5])
@pytest.mark.parametrize("n_targets", [1, 5])
@pytest.mark.parametrize("freq", ["T"])
def test_handle_features_dataset_remove_dynamic_cat(
    dataset_type, np_rng, n_groups, n_targets, freq
):
    (
        df_dict,
        target_columns,
        real_dynamic_feature_columns,
        cat_dynamic_feature_columns,
    ) = generate_forecast_df_dict(
        np_rng,
        n_groups,
        n_targets=n_targets,
        freq=freq,
        n_real_features=np_rng.integers(1, 10),
        n_cat_features=np_rng.integers(1, 10),
    )
    gluonts_freq = find_gluonts_freq(freq)

    real_static_feature_dict = {}
    for group in df_dict.keys():
        n_features = np_rng.integers(2, 10)
        real_static_feature_dict[group] = np_rng.standard_normal(n_features)

    cat_static_feature_dict = {}
    for group in df_dict.keys():
        n_features = np_rng.integers(2, 10)
        cat_static_feature_dict[group] = np_rng.standard_normal(n_features)

    original_dataset = dataframe_to_list_dataset(
        df_dict,
        target_columns,
        gluonts_freq,
        real_static_feature_dict=real_static_feature_dict,
        cat_static_feature_dict=cat_static_feature_dict,
        real_dynamic_feature_columns=real_dynamic_feature_columns,
        cat_dynamic_feature_columns=cat_dynamic_feature_columns,
    )

    if dataset_type == "transformed_dataset":
        original_dataset = TransformedDataset(original_dataset, None)

    keep_feat_static_real = True
    keep_feat_static_cat = True
    keep_feat_dynamic_real = True
    keep_feat_dynamic_cat = False

    clean_dataset = handle_features_dataset(
        original_dataset,
        keep_feat_static_real,
        keep_feat_static_cat,
        keep_feat_dynamic_real,
        keep_feat_dynamic_cat,
    )

    if dataset_type == "transformed_dataset":
        original_list_data = original_dataset.base_dataset.list_data
        clean_list_data = clean_dataset.base_dataset.list_data
    else:
        original_list_data = original_dataset.list_data
        clean_list_data = clean_dataset.list_data

    assert len(clean_list_data) == len(df_dict)

    for group_index, (group_name, df_group) in enumerate(df_dict.items()):
        original_data = original_list_data[group_index]
        data = clean_list_data[group_index]

        assert data["start"] == df_group.index[0]

        if n_targets > 1:
            for target_index, target in enumerate(target_columns):
                assert (data["target"][target_index, :] == df_group[target]).all()
        else:
            assert (data["target"] == df_group[target_columns[0]]).all()

        assert "feat_dynamic_cat" not in data
        for feat_index, feat_column in enumerate(cat_dynamic_feature_columns):
            assert (
                original_data["feat_dynamic_cat"][feat_index, :]
                == df_group[feat_column]
            ).all()

        for feat in real_static_feature_dict[group_name]:
            assert feat in data["feat_static_real"]
        for feat in cat_static_feature_dict[group_name]:
            assert feat in data["feat_static_cat"]
        for feat_index, feat_column in enumerate(real_dynamic_feature_columns):
            assert (
                data["feat_dynamic_real"][feat_index, :] == df_group[feat_column]
            ).all()
