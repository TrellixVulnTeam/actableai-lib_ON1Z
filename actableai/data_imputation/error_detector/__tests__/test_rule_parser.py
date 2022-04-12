from unittest.mock import MagicMock, patch, call

from actableai.data_imputation.error_detector.rule_parser import RulesBuilder, RulesRaw
from actableai.data_imputation.type_recon.type_detector import DfTypes


class TestRulesBuilderParser:
    @patch("actableai.data_imputation.error_detector.rule_parser.Constraints.parse")
    @patch("actableai.data_imputation.error_detector.rule_parser.MatchRules.parse")
    def test_construct(self, mock_match_rules_parse, mock_constraints_parse):
        mock_df_types = MagicMock(DfTypes)
        mock_rules = MagicMock(RulesRaw)
        mock_validations = MagicMock()
        mock_misplaced = MagicMock()
        mock_rules.validations = mock_validations
        mock_rules.misplaced = mock_misplaced

        RulesBuilder.parse(mock_df_types, mock_rules)
        assert mock_constraints_parse.call_count == 1
        assert mock_match_rules_parse.call_count == 1
        assert mock_constraints_parse.call_args_list == [call(mock_validations)]
        assert mock_match_rules_parse.call_args_list == [
            call(mock_df_types, mock_misplaced)
        ]
