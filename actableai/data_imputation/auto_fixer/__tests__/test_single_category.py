import numpy as np
import pandas as pd
import pytest

from actableai.data_imputation.auto_fixer.fix_info import FixInfoList, FixInfo
from actableai.data_imputation.auto_fixer.single_category_fixer import (
    SingleCategoryFixer,
)
from actableai.data_imputation.error_detector.cell_erros import (
    CellErrors,
    ErrorType,
    CellError,
)
from actableai.data_imputation.meta.column import RichColumnMeta
from actableai.data_imputation.meta.types import ColumnType
from actableai.data_imputation.type_recon.type_detector import DfTypes


@pytest.mark.parametrize(
    "df, errors, current_column, expect_fix_info_list",
    [
        (
            pd.DataFrame({"a": [np.nan, 1, None, np.nan, None]}),
            CellErrors(
                DfTypes([("a", ColumnType.Integer)]),
                [
                    CellError("a", 0, ErrorType.NULL),
                    CellError("a", 2, ErrorType.NULL),
                    CellError("a", 3, ErrorType.NULL),
                    CellError("a", 4, ErrorType.NULL),
                ],
            ),
            RichColumnMeta("a", ColumnType.Integer),
            FixInfoList(
                [
                    FixInfo("a", 0, 1.0),
                    FixInfo("a", 2, 1.0),
                    FixInfo("a", 3, 1.0),
                    FixInfo(
                        "a",
                        4,
                        1,
                    ),
                ]
            ),
        )
    ],
)
def test_fix(df, errors, current_column, expect_fix_info_list):
    fixer = SingleCategoryFixer()
    fix_info_list = fixer.fix(df, errors, current_column)

    assert fix_info_list == expect_fix_info_list
