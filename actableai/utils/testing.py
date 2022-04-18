import itertools
import random
import datetime
from copy import deepcopy

import psutil
import ray

import numpy as np
import pandas as pd


def unittest_hyperparameters():
    return {"RF": {}}


def init_ray(**kwargs):
    num_cpus = psutil.cpu_count()
    ray.init(
        num_cpus=num_cpus,
        namespace="aai",
        **kwargs
    )


def generate_random_date(np_rng=None,
                         min_year=1900,
                         min_month=1,
                         min_day=1,
                         max_year=2000,
                         max_month=1,
                         max_day=1,
                         random_state=None):
    """
    TODO write documentation
    """
    min_date = datetime.date(min_year, min_month, min_day)
    max_date = datetime.date(max_year, max_month, max_day)

    max_days = (max_date - min_date).days

    days_delta = None
    if np_rng is None:
        days_delta = random.randrange(max_days)
    else:
        days_delta = np_rng.integers(max_days)

    return min_date + datetime.timedelta(days=int(days_delta))


def generate_date_range(np_rng=None,
                        start_date=None,
                        min_periods=10,
                        max_periods=60,
                        periods=None,
                        freq=None):
    """
    TODO write documentation
    """

    if start_date is None:
        start_date = generate_random_date(np_rng)
    if periods is None:
        if np_rng is None:
            periods = random.randrange(min_periods, max_periods)
        else:
            periods = np_rng.integers(min_periods, max_periods)
    if freq is None:
        freq = "T"

    return pd.date_range(start_date, periods=periods, freq=freq)


def generate_forecast_df_dict(np_rng,
                              n_groups=1,
                              n_targets=1,
                              freq=None,
                              n_real_features=0,
                              n_cat_features=0,
                              date_range_kwargs=None):
    """
    TODO write documentation
    """
    if date_range_kwargs is None:
        date_range_kwargs = {}

    df_dict = {}
    target_list = [f"target_{i}" for i in range(n_targets)]
    real_feature_list = [f"real_feat_{i}" for i in range(n_real_features)]
    cat_feature_list = [f"cat_feat_{i}" for i in range(n_cat_features)]

    for i in range(n_groups):
        date_range = generate_date_range(np_rng, freq=freq, **date_range_kwargs)

        data = {}
        for target in target_list:
            data[target] = np_rng.standard_normal(len(date_range))
        for feat in real_feature_list:
            data[feat] = np_rng.standard_normal(len(date_range))
        for feat in cat_feature_list:
            data[feat] = np_rng.integers(1, 10, len(date_range))

        df = pd.DataFrame(data)
        df.index = date_range
        df = df.sort_index()

        df_dict[f"group_{i}"] = df

    return df_dict, target_list, real_feature_list, cat_feature_list


def generate_forecast_df(np_rng,
                         prediction_length,
                         n_group_by=0,
                         n_targets=1,
                         freq=None,
                         n_real_static_features=0,
                         n_cat_static_features=0,
                         n_real_dynamic_features=0,
                         n_cat_dynamic_features=0,
                         date_range_kwargs=None):
    """
    TODO write documentation
    """
    has_groups = n_group_by > 0
    group_values_list = []

    n_groups = 1
    if has_groups:
        for group_by_index in range(n_group_by):
            n_group_cat = np_rng.integers(2, 5)
            n_groups *= n_group_cat

            group_values_list.append([f"group_val_{i}" for i in range(n_group_cat)])

    group_by = [f"group_{i}" for i in range(n_group_by)]
    real_static_feature_list = [f"real_static_feat_{i}" for i in range(n_real_static_features)]
    cat_static_feature_list = [f"cat_static_feat_{i}" for i in range(n_cat_static_features)]

    df_dict, target_list, real_dynamic_feature_list, cat_dynamic_feature_list = generate_forecast_df_dict(
        np_rng,
        n_groups=n_groups,
        n_targets=n_targets,
        freq=freq,
        n_real_features=n_real_dynamic_features,
        n_cat_features=n_cat_dynamic_features,
        date_range_kwargs=date_range_kwargs
    )

    feature_list = real_static_feature_list \
        + cat_static_feature_list \
        + real_dynamic_feature_list \
        + cat_dynamic_feature_list

    df = pd.DataFrame()

    for (group, df_group), group_values in zip(df_dict.items(), itertools.product(*group_values_list)):
        df_group["date"] = df_group.index

        if has_groups:
            for group_by_name, group_by_value in zip(group_by, group_values):
                df_group[group_by_name] = group_by_value

        for real_static_feat in real_static_feature_list:
            df_group[real_static_feat] = np_rng.standard_normal()
        for cat_static_feat in cat_static_feature_list:
            df_group[cat_static_feat] = np_rng.integers(1, 10)

        if len(feature_list) > 0:
            df_group.loc[-prediction_length:, target_list] = np.nan

        df = pd.concat([
            df,
            df_group
        ], ignore_index=True)

    return df, \
           "date", \
           target_list, \
           group_by, \
           feature_list, \
           n_groups

