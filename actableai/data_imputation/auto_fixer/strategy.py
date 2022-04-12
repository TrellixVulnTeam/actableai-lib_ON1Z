from collections import Counter
from enum import Enum, auto

import pandas as pd

from actableai.data_imputation import config
from actableai.data_imputation.auto_fixer.auto_gluon_fixer import AutoGluonFixer
from actableai.data_imputation.auto_fixer.datetime_fixer import DatetimeFixer
from actableai.data_imputation.auto_fixer.neighbor_fixer import NeighborFixer
from actableai.data_imputation.auto_fixer.single_category_fixer import (
    SingleCategoryFixer,
)
from actableai.data_imputation.error_detector import ColumnErrors
from actableai.data_imputation.meta.types import ColumnType


class FixStrategy(Enum):
    AUTO = auto()
    NEIGHBOR = auto()
    AUTOGLUON = auto()
    SINGLE_CATEGORY = auto()
    UNDECIDED = auto()
    UNABLE_TO_FIX = auto()
    DATETIME = auto()


def get_fixer(strategy: FixStrategy):
    if strategy == FixStrategy.SINGLE_CATEGORY:
        return SingleCategoryFixer()
    elif strategy == FixStrategy.NEIGHBOR:
        return NeighborFixer()
    elif strategy == FixStrategy.AUTOGLUON:
        return AutoGluonFixer()
    elif strategy == FixStrategy.DATETIME:
        return DatetimeFixer()

    raise NotImplementedError


def get_quick_fixer_for_debug(column_type: ColumnType):
    if column_type == ColumnType.Category:
        return SingleCategoryFixer()
    else:
        return NeighborFixer()


def determine_fix_strategy(
    series: pd.Series, column_type: ColumnType, errors: ColumnErrors
) -> FixStrategy:
    remain_series = series[~series.index.isin([err.index for err in errors])]

    counter = Counter(remain_series)
    if len(counter) == 0:
        return FixStrategy.UNABLE_TO_FIX
    value_count = counter.most_common(1)[0][1]
    if column_type == ColumnType.Timestamp:
        return FixStrategy.DATETIME
    elif (
        value_count < config.UNABLE_TO_FIX_DISTINCT_SIZE_THRESHOLD
        and column_type == ColumnType.Category
    ):
        return FixStrategy.SINGLE_CATEGORY

    elif (
        sum(pd.isnull(remain_series)) == len(remain_series)
        or column_type == ColumnType.NULL
    ):
        return FixStrategy.UNABLE_TO_FIX
    elif len(set(remain_series[~pd.isnull(remain_series)])) == 1:
        return FixStrategy.SINGLE_CATEGORY
    elif (
        len(set(remain_series[~pd.isnull(remain_series)])) <= 2
        or len(remain_series) < config.SMALL_DATASET_LENGTH_THRESHOLD
    ):
        return FixStrategy.NEIGHBOR
    else:
        return FixStrategy.AUTOGLUON


def determine_refine_strategy(series: pd.Series, errors: ColumnErrors) -> FixStrategy:
    remain_series = series[~series.index.isin([err.index for err in errors])]

    if len(set(remain_series[~pd.isnull(remain_series)])) == 1:
        return FixStrategy.SINGLE_CATEGORY
    else:
        return FixStrategy.AUTOGLUON
