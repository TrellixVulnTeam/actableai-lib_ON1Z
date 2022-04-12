import numpy as np
import pandas as pd

from actableai.data_imputation.auto_fixer.auto_fixer import AutoFixer
from actableai.data_imputation.auto_fixer.fix_info import (
    FixInfoList,
    FixInfo,
    FixValueOptions,
    FixValue,
)
from actableai.data_imputation.error_detector import CellErrors
from actableai.data_imputation.meta.column import RichColumnMeta


class SingleCategoryFixer(AutoFixer):
    def fix(
        self,
        df: pd.DataFrame,
        all_errors: CellErrors,
        current_column: RichColumnMeta,
    ) -> FixInfoList:
        fix_info_list = FixInfoList()

        category_fix_value = None
        all_category = set(df[current_column.name])

        while (
            (category_fix_value is None)
            or (category_fix_value == "nan")
            or (isinstance(category_fix_value, float) and np.isnan(category_fix_value))
        ):
            category_fix_value = all_category.pop()

        for err in all_errors[current_column.name]:
            fix_info_list.append(
                FixInfo(
                    col=err.column,
                    index=err.index,
                    options=FixValueOptions(options=[FixValue(category_fix_value, 1)]),
                )
            )

        return fix_info_list
