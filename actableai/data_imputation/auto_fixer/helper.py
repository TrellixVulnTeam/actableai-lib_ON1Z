from typing import Union

import numpy as np
import pandas as pd

from actableai.data_imputation.auto_fixer.fix_info import FixInfoList, FixInfo
from actableai.data_imputation.error_detector import (
    CellErrors,
    ColumnErrors,
)
from actableai.data_imputation.meta.column import NumWithTagColumnMeta
from actableai.data_imputation.type_recon.type_detector import DfTypes


def get_df_without_error(
    df: pd.DataFrame, errors: Union[CellErrors, ColumnErrors]
) -> pd.DataFrame:
    indexes = set()
    for error in errors:
        indexes.add(error.index)
    return df[~df.index.isin(indexes)]


def get_df_with_only_error(
    df: pd.DataFrame, errors: Union[CellErrors, ColumnErrors]
) -> pd.DataFrame:
    indexes = set()
    for error in errors:
        indexes.add(error.index)
    return df[df.index.isin(indexes)]


def fulfil_fix_back(
    df_to_fix: pd.DataFrame, fix_info_list: FixInfoList
) -> pd.DataFrame:
    fix_info: FixInfo
    for fix_info in fix_info_list:
        df_to_fix.at[fix_info.index, fix_info.col] = fix_info.best_guess

    return df_to_fix


def merge_num_with_tag_columns(
    df_fixed: pd.DataFrame, column_types: DfTypes
) -> pd.DataFrame:
    df = pd.DataFrame()
    for col in column_types.columns_original:
        col_meta = column_types.get_meta(col)
        if isinstance(col_meta, NumWithTagColumnMeta):
            num_column = df_fixed[col_meta.get_num_column_name()]
            if col_meta.num_type == "int":
                num_column = num_column.apply(lambda x: "" if pd.isna(x) else int(x))
            df[col_meta.name] = (
                df_fixed[col_meta.get_left_tag_column_name()].astype(str)
                + num_column.astype(str)
                + df_fixed[col_meta.get_right_tag_column_name()].astype(str)
            )
            is_na = df_fixed[col_meta.get_num_column_name()].isna()
            if any(is_na):
                df[col_meta.name][is_na] = np.nan
        else:
            df[col] = df_fixed[col]
    return df


def finalize_columns(original_df: pd.DataFrame, fixed_df: pd.DataFrame) -> pd.DataFrame:
    """
    Use columns in fixed_df to replace original_df columns, due to original_df might have some unsupported columns
    :param original_df: the original df waiting to fix
    :param fixed_df: the fixed df, but might have some columns missing from original_df due to they are not support to
        be fixed
    :return: the complete dataframe
    """
    final_df = original_df.copy()
    final_df[fixed_df.columns] = fixed_df
    return final_df
