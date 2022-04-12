from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from actableai.data_imputation.error_detector.cell_erros import (
    ErrorCandidate,
    ErrorColumns,
    CellError,
    ErrorType,
)
from actableai.data_imputation.error_detector.constraint import Constraints
from actableai.data_imputation.error_detector.validation_constraints_checker import (
    ValidationConstrainsChecker,
)

stub_path = "actableai.data_imputation.error_detector.validation_constraints_checker"


@pytest.fixture(autouse=True)
def df():
    return pd.DataFrame.from_dict(
        {
            "first_set": [
                1,
                1,
                2,
                2,
                3,
            ],
            "second_set": [
                "a",
                "a",
                "b",
                "c",
                "a",
            ],
            "third set": [
                "X",
                "Y",
                "Z",
                "Z",
                "Y",
            ],
        }
    )


class TestFindAllUnmatched:
    @pytest.mark.parametrize(
        "constraints,expect_call_count,expect_sql",
        [
            (
                "first_set=first_set -> second_set<>second_set & first_set<second_set",
                1,
                [
                    "SELECT DISTINCT t1.id FROM df as t1, df as t2 "
                    "WHERE t1.id<>t2.id AND t1.`first_set`=t2.`first_set` "
                    "AND t1.`second_set`<>t2.`second_set` AND t1.`first_set`<t2.`second_set`"
                ],
            ),
            (
                "first_set=first_set -> second_set<>second_set & first_set<second_set "
                "OR first_set>first_set-> second_set=first_set",
                2,
                [
                    "SELECT DISTINCT t1.id FROM df as t1, df as t2 "
                    "WHERE t1.id<>t2.id AND t1.`first_set`=t2.`first_set` "
                    "AND t1.`second_set`<>t2.`second_set` AND t1.`first_set`<t2.`second_set`",
                    "SELECT DISTINCT t1.id FROM df as t1, df as t2 "
                    "WHERE t1.id<>t2.id AND t1.`first_set`>t2.`first_set` AND t1.`second_set`=t2.`first_set`",
                ],
            ),
            (
                "first_set=first_set -> second_set<>second_set & third set<third set ",
                1,
                [
                    "SELECT DISTINCT t1.id FROM df as t1, df as t2 "
                    "WHERE t1.id<>t2.id AND t1.`first_set`=t2.`first_set` "
                    "AND t1.`second_set`<>t2.`second_set` AND t1.`third set`<t2.`third set`",
                ],
            ),
        ],
    )
    @patch(f"{stub_path}.ps")
    def test_generate_detector_sql(
        self, mock_ps, df, constraints, expect_call_count, expect_sql
    ):
        mock_ps.return_value = pd.DataFrame.from_dict({"id": [1, 2, 3]})

        detector = ValidationConstrainsChecker(Constraints.parse(constraints))
        detector.find_all_unmatched(df)
        assert mock_ps.sqldf.call_count == expect_call_count
        assert [
            mock_ps.sqldf.call_args_list[i][0][0] for i in range(expect_call_count)
        ] == expect_sql

    @pytest.mark.parametrize(
        "constraints,expect_errors",
        [
            (
                "first_set=first_set -> second_set<>second_set",
                [
                    ErrorCandidate(
                        index=2,
                        columns=ErrorColumns({"first_set"}, {"second_set"}),
                    ),
                    ErrorCandidate(
                        index=3,
                        columns=ErrorColumns({"first_set"}, {"second_set"}),
                    ),
                ],
            ),
            (
                "second_set=second_set -> first_set=first_set OR first_set<>first_set -> second_set=second_set",
                [
                    ErrorCandidate(
                        index=0,
                        columns=ErrorColumns(
                            {"first_set", "second_set"},
                            {"first_set", "second_set"},
                        ),
                    ),
                    ErrorCandidate(
                        index=1,
                        columns=ErrorColumns(
                            {"first_set", "second_set"},
                            {"first_set", "second_set"},
                        ),
                    ),
                    ErrorCandidate(
                        index=4,
                        columns=ErrorColumns(
                            {"first_set"},
                            {"second_set"},
                        ),
                    ),
                ],
            ),
            (
                "first_set=first_set & second_set <> second_set-> third set = third set",
                [
                    ErrorCandidate(
                        index=2,
                        columns=ErrorColumns(
                            {"first_set", "second_set"},
                            {"third set"},
                        ),
                    ),
                    ErrorCandidate(
                        index=3,
                        columns=ErrorColumns(
                            {"first_set", "second_set"},
                            {"third set"},
                        ),
                    ),
                ],
            ),
        ],
    )
    def test_detect_invalid_candidates(self, df, constraints, expect_errors):
        detector = ValidationConstrainsChecker(Constraints.parse(constraints))
        errors = detector.find_all_unmatched(df)
        assert len(errors) == len(expect_errors)
        errors = iter(sorted(errors, key=lambda x: f"{x.index}"))
        for idx, error in enumerate(errors):
            assert error == expect_errors[idx], f"{idx} not match"


class TestDetectTypoCells:
    def test_detect(self):
        df = pd.DataFrame(
            data={
                "HospitalName": ["callahan eye foundation hospital"] * 5
                + [
                    "marshall medical center south",
                    "marshall medical center south",
                    "marshal medical centxr south",
                    "marshall medical center south",
                ]
                + ["some other hospital"],
                "Address1": ["1720 university blvd"] * 3
                + ["1720 univxrsity blvd", "1720 university blvd"]
                + [
                    "2505 u s highway 431 north",
                    "2505 u s highway 431 north",
                    "2505 u s highway 431 north",
                    "2505 u s highway 431 north",
                ]
                + ["1720 university blvd"],
            }
        )

        constraints = MagicMock(Constraints)
        detector = ValidationConstrainsChecker(constraints)
        constraints.mentioned_columns = ["HospitalName", "Address1"]
        mock_func = MagicMock()
        detector.find_all_unmatched = mock_func
        mock_func.return_value = [
            ErrorCandidate(i, ErrorColumns({"HospitalName"}, {"Address1"}))
            for i in range(df.shape[0])
        ]

        errors = detector.detect_typo_cells(df)
        assert errors == [
            CellError("Address1", 3, ErrorType.TYPO),
            CellError("HospitalName", 7, ErrorType.TYPO),
            CellError("HospitalName", 9, ErrorType.TYPO),
        ]
