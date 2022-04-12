import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer

from actableai.data_imputation.auto_fixer.auto_fixer import AutoFixer
from actableai.data_imputation.auto_fixer.errors import EmptyTrainDataException
from actableai.data_imputation.auto_fixer.fix_info import (
    FixInfoList,
    FixInfo,
    FixValueOptions,
    FixValue,
)
from actableai.data_imputation.auto_fixer.helper import get_df_without_error

from actableai.data_imputation.error_detector import CellErrors
from actableai.data_imputation.meta.column import RichColumnMeta


class NeighborFixer(AutoFixer):
    def __init__(self):
        self._imp = IterativeImputer(max_iter=20, random_state=0)
        self.__cached_matrix_after_fit = None

    def fix(
        self,
        df: pd.DataFrame,
        all_errors: CellErrors,
        current_column: RichColumnMeta,
    ) -> FixInfoList:
        df_to_train = get_df_without_error(df, all_errors[current_column.name])
        if df_to_train.empty:
            raise EmptyTrainDataException()

        if self.__cached_matrix_after_fit is None:
            df_to_matrix = df.select_dtypes(exclude=["datetime"]).to_numpy()
            self.__cached_matrix_after_fit = self._imp.fit_transform(df_to_matrix)

        column_index_to_fix = df.select_dtypes(exclude=["datetime"]).columns.get_loc(
            current_column.name
        )

        fix_info_list = FixInfoList()
        for err in all_errors[current_column.name]:
            options = FixValueOptions(
                options=[
                    FixValue(
                        value=self.__cached_matrix_after_fit[err.index][
                            column_index_to_fix
                        ],
                        confidence=1,
                    )
                ]
            )
            fix_info_list.append(
                FixInfo(
                    col=current_column.name,
                    index=err.index,
                    options=options,
                )
            )
        return fix_info_list
