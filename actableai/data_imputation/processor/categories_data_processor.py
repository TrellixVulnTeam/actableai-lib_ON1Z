import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

from actableai.data_imputation.config.config import NAN_REPLACE_VALUE
from actableai.data_imputation.meta.types import ColumnType
from actableai.data_imputation.type_recon.type_detector import DfTypes


class CategoriesDataProcessor:
    def __init__(self):
        self.__ordinal_encoder = OrdinalEncoder()
        self.__categories_columns = []
        self.__non_categories_columns = []

    def encode(self, df: pd.DataFrame, column_types: DfTypes) -> pd.DataFrame:
        """
        Encode data into categories number
        we assume the data in categories column are all string, this is handled in `convert_df_for_fix`
        """
        columns = df.columns
        for col in column_types.columns_to_fix:
            if column_types[col] == ColumnType.Category:
                self.__categories_columns.append(col)
            else:
                self.__non_categories_columns.append(col)

        df_only_category = df[self.__categories_columns].copy()
        df_only_category.fillna(NAN_REPLACE_VALUE, inplace=True)

        self.__ordinal_encoder.fit(df_only_category)
        predict_result = self.__ordinal_encoder.transform(df_only_category)
        df_result = pd.DataFrame(
            data=dict(zip(self.__categories_columns, predict_result.T))
        )
        df_result[self.__non_categories_columns] = df[self.__non_categories_columns]
        return df_result[columns]

    def decode(self, df: pd.DataFrame) -> pd.DataFrame:
        columns = df.columns
        if len(self.__categories_columns) == 0:
            return df

        df[self.__categories_columns] = (
            df[self.__categories_columns].apply(round).astype("int")
        )

        for i, col in enumerate(self.__categories_columns):
            min_value, max_value = (
                0,
                len(self.__ordinal_encoder.categories_[i]) - 1,
            )
            series = df[col]
            series.where(series >= min_value, min_value, inplace=True)
            series.where(series <= max_value, max_value, inplace=True)

            categories = self.__ordinal_encoder.categories_[i].tolist()
            if NAN_REPLACE_VALUE in categories:
                nan_index = categories.index(NAN_REPLACE_VALUE)
                nan_replace_value = nan_index + 1
                series.where(series != nan_index, nan_replace_value, inplace=True)
            df[col] = series

        df_result = pd.DataFrame(
            data=dict(
                zip(
                    self.__categories_columns,
                    self.__ordinal_encoder.inverse_transform(
                        df[self.__categories_columns]
                    ).T,
                )
            )
        )
        df_result[self.__non_categories_columns] = df[self.__non_categories_columns]

        return df_result[columns]
