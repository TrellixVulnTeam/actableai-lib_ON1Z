from unittest.mock import MagicMock, call

import pandas as pd
import pytest

from actableai.data_imputation.error_detector.column_format import (
    MatchStrRule,
    PresetRuleName,
    MatchRules,
    MatchNumRule,
)
from actableai.data_imputation.error_detector.match_condition import ConditionOp
from actableai.data_imputation.error_detector.misplaced_detector import (
    MisplacedDetector,
)
from actableai.data_imputation.meta import ColumnType
from actableai.data_imputation.type_recon.regex_consts import REGEX_CONSTS
from actableai.data_imputation.type_recon.type_detector import DfTypes


class TestMisplacedDetector:
    @pytest.mark.parametrize(
        "df, dftypes, preset_rules, customize_rules, expect_format",
        [
            (
                pd.DataFrame(data={"a": []}),
                DfTypes([("a", ColumnType.Percentage)]),
                [PresetRuleName.SmartPercentage],
                MatchRules([]),
                [
                    MatchStrRule(
                        column="a",
                        match_str=REGEX_CONSTS[ColumnType.Percentage],
                        is_regex=True,
                        op=ConditionOp.IQ,
                    )
                ],
            ),
            (
                pd.DataFrame(data={"a": []}),
                DfTypes([("a", ColumnType.Temperature)]),
                [PresetRuleName.SmartTemperature],
                MatchRules([]),
                [
                    MatchStrRule(
                        column="a",
                        match_str=REGEX_CONSTS[ColumnType.Temperature],
                        is_regex=True,
                        op=ConditionOp.IQ,
                    )
                ],
            ),
            (
                pd.DataFrame(data={"a": [], "b": []}),
                DfTypes([("a", ColumnType.Temperature), ("b", ColumnType.String)]),
                [PresetRuleName.SmartTemperature],
                MatchRules(
                    [
                        MatchStrRule(
                            column="b",
                            match_str="",
                            is_regex=False,
                            op=ConditionOp.EQ,
                        )
                    ]
                ),
                [
                    MatchStrRule(
                        column="b",
                        match_str="",
                        is_regex=False,
                        op=ConditionOp.EQ,
                    ),
                    MatchStrRule(
                        column="a",
                        match_str=REGEX_CONSTS[ColumnType.Temperature],
                        is_regex=True,
                        op=ConditionOp.IQ,
                    ),
                ],
            ),
        ],
    )
    def test_construct(self, df, dftypes, preset_rules, customize_rules, expect_format):
        detector = MisplacedDetector(
            preset_rules=preset_rules, customize_rules=customize_rules
        )
        detector.setup(df, dftypes)
        assert [rule for _, rule in list(detector._rules)] == expect_format

    def test_detect_cells(self):
        df = MagicMock(pd.DataFrame)
        series_a = MagicMock(pd.Series)
        series_b = MagicMock(pd.Series)
        df.__getitem__.side_effect = [series_a, series_b]
        dftypes = MagicMock(DfTypes)
        rule1 = MagicMock(MatchStrRule)
        rule1.column = "a"
        rule2 = MagicMock(MatchNumRule)
        rule2.column = "b"
        rules = MatchRules([rule1, rule2])

        detector = MisplacedDetector(customize_rules=rules)
        detector.setup(df, dftypes)

        detector.detect_cells()

        assert df.__getitem__.call_args_list == [
            call(rule1.column),
            call(rule2.column),
        ]
        assert rule1.find_misplaced.call_count == 1
        assert rule1.find_misplaced.call_args == call(series_a)
        assert rule2.find_misplaced.call_count == 1
        assert rule2.find_misplaced.call_args == call(series_b)
