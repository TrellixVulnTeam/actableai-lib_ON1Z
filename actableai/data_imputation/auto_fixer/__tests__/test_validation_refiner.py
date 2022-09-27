import pandas as pd
import numpy as np
import pytest
from math import inf
from unittest.mock import MagicMock

from actableai.data_imputation.auto_fixer.__tests__.helper import (
    assert_fix_info_list,
)
from actableai.data_imputation.auto_fixer.fix_info import (
    FixInfo,
    FixValueOptions,
    FixValue,
    FixInfoList,
)
from actableai.data_imputation.auto_fixer.validation_refiner import ValidationRefiner
from actableai.data_imputation.error_detector import ErrorDetector, CellErrors
from actableai.data_imputation.error_detector.cell_erros import (
    CellError,
    ErrorType,
)
from actableai.data_imputation.error_detector.constraint import Constraints

stub = "actableai.data_imputation.auto_fixer.refiner"


class TestFindAllValuePairsSatisfyConstraints:
    def test_should_find(self):
        mock_constraints = MagicMock(Constraints)
        mock_constraints.mentioned_columns = ["a", "b"]

        mock_error_detector = MagicMock(ErrorDetector)
        errors = CellErrors(
            MagicMock(),
            [
                CellError("a", 1, ErrorType.INVALID),
                CellError("b", 3, ErrorType.INVALID),
            ],
        )

        df = pd.DataFrame(
            {"a": [1, 2, 3, 7, 1], "b": [4, 5, 6, 8, 4], "c": [1, 1, 1, 1, 1]}
        )
        df_expect = pd.DataFrame({"a": [1, 3], "b": [4, 6]})

        refiner = ValidationRefiner(mock_error_detector)
        refiner._custom_constraints = mock_constraints
        result = refiner._find_all_value_pairs_satisfy_constraints(errors, df)

        assert result.equals(df_expect)


class TestGetBestFixPair:
    @pytest.mark.parametrize(
        "fix_info_for_row, index, correlations, possible_value_pairs, expect",
        [
            (
                FixInfoList([]),
                1,
                pd.DataFrame({"a": [1, 0.3], "b": [0.3, 1]}, index=["a", "b"]),
                pd.DataFrame({"a": ["a fix"], "b": ["b value"]}),
                [],
            ),
            (
                FixInfoList(
                    [
                        FixInfo(
                            col="a",
                            index=1,
                            options=FixValueOptions(
                                [
                                    FixValue(value="a fix", confidence=0.8),
                                    FixValue(value="a fix fake", confidence=0.2),
                                ]
                            ),
                        )
                    ]
                ),
                1,
                pd.DataFrame({"a": [1, 0.3], "b": [0.3, 1]}, index=["a", "b"]),
                pd.DataFrame(),
                [],
            ),
            (
                FixInfoList(
                    [
                        FixInfo(
                            col="a",
                            index=1,
                            options=FixValueOptions(
                                [
                                    FixValue(value="a fix", confidence=0.8),
                                    FixValue(value="a fix fake", confidence=0.2),
                                ]
                            ),
                        )
                    ]
                ),
                1,
                pd.DataFrame({"a": [1, 0.3], "b": [0.3, 1]}, index=["a", "b"]),
                pd.DataFrame({"a": ["a fix"], "b": ["b value"]}),
                [
                    FixInfo(
                        col="a",
                        index=1,
                        options=FixValueOptions(
                            [
                                FixValue(value="a fix", confidence=0.8),
                            ]
                        ),
                    ),
                ],
            ),
            (
                FixInfoList(
                    [
                        FixInfo(
                            col="a",
                            index=1,
                            options=FixValueOptions(
                                [
                                    FixValue(value="a fix", confidence=0.8),
                                    FixValue(value="a fix fake", confidence=0.2),
                                ]
                            ),
                        ),
                        FixInfo(
                            col="b",
                            index=1,
                            options=FixValueOptions(
                                [
                                    FixValue(value="b fix fake", confidence=0.4),
                                    FixValue(value="b fix", confidence=0.6),
                                ]
                            ),
                        ),
                    ]
                ),
                1,
                pd.DataFrame({"a": [1, 0.3], "b": [0.3, 1]}, index=["a", "b"]),
                pd.DataFrame({"a": ["a fix"], "b": ["b fix"]}),
                [
                    FixInfo(
                        col="a",
                        index=1,
                        options=FixValueOptions(
                            [
                                FixValue(value="a fix", confidence=0.8),
                            ]
                        ),
                    ),
                    FixInfo(
                        col="b",
                        index=1,
                        options=FixValueOptions(
                            [
                                FixValue(value="b fix", confidence=0.6),
                            ]
                        ),
                    ),
                ],
            ),
            (
                FixInfoList(
                    [
                        FixInfo(
                            col="a",
                            index=1,
                            options=FixValueOptions(
                                [
                                    FixValue(value="a fix", confidence=0.8),
                                    FixValue(value="a fix fake", confidence=0.2),
                                ]
                            ),
                        ),
                        FixInfo(
                            col="b",
                            index=1,
                            options=FixValueOptions(
                                [
                                    FixValue(value="b fix fake", confidence=0.4),
                                    FixValue(value="b fix", confidence=0.6),
                                ]
                            ),
                        ),
                    ]
                ),
                1,
                pd.DataFrame(
                    {
                        "a": [1, 0.2, 0.3],
                        "b": [0.3, 1, 0.4],
                        "c": [0.3, 0.4, 1],
                    },
                    index=["a", "b", "c"],
                ),
                pd.DataFrame({"a": ["a fix"], "b": ["b fix"], "c": ["c fix"]}),
                [
                    FixInfo(
                        col="a",
                        index=1,
                        options=FixValueOptions(
                            [
                                FixValue(value="a fix", confidence=0.8),
                            ]
                        ),
                    ),
                    FixInfo(
                        col="b",
                        index=1,
                        options=FixValueOptions(
                            [
                                FixValue(value="b fix", confidence=0.6),
                            ]
                        ),
                    ),
                ],
            ),
        ],
    )
    def test_calc(
        self,
        fix_info_for_row,
        index,
        correlations,
        possible_value_pairs,
        expect,
    ):
        refiner = ValidationRefiner(MagicMock())
        result = refiner._get_best_fix_pair(
            fix_info_for_row, index, correlations, possible_value_pairs
        )
        assert_fix_info_list(result, expect)


class TestReplaceAsUnableToFixInPlace:
    @pytest.mark.parametrize(
        "errors, df, fix_info_list, expect_df, expect_fix_info_list",
        [
            (
                CellErrors(MagicMock(), [CellError("a", 0, ErrorType.INVALID)]),
                pd.DataFrame({"a": [1, 2, 3]}),
                FixInfoList([FixInfo("a", 0, FixValueOptions([FixValue(100, 1)]))]),
                pd.DataFrame({"a": [np.nan, 2, 3]}),
                FixInfoList(
                    [
                        FixInfo(
                            "a",
                            0,
                            FixValueOptions([FixValue(np.nan, -inf)]),
                        )
                    ]
                ),
            )
        ],
    )
    def test_should_replace_df_and_fix_info_list(
        self, errors, df, fix_info_list, expect_df, expect_fix_info_list
    ):
        refiner = ValidationRefiner(MagicMock())
        refiner._replace_as_unable_to_fix_in_place(errors, df, fix_info_list)

        assert df.equals(expect_df)
        assert fix_info_list == expect_fix_info_list
