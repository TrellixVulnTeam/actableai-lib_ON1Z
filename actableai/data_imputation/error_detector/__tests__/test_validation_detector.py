from unittest.mock import patch, MagicMock, call

import pandas as pd
import pytest

from actableai.data_imputation.error_detector.constraint import (
    Constraints,
)
from actableai.data_imputation.error_detector.validation_detector import (
    ValidationDetector,
)
from actableai.data_imputation.error_detector.cell_erros import (
    ErrorColumns,
    ErrorCandidate,
    CellErrors,
)
from actableai.data_imputation.type_recon.type_detector import DfTypes
from actableai.data_imputation.meta.types import ColumnType


stub_path = "actableai.data_imputation.error_detector.validation_detector"


class TestDetectCell:
    @patch(f"{stub_path}.SmartErrorCellSelector")
    @patch(f"{stub_path}.ValidationConstrainsChecker")
    def test_detect_should_call_detect_candidate_and_find_actual_errors(
        self, validation_checker_mod, smart_selector_mod
    ):
        mock_detect = MagicMock()
        mock_selector = MagicMock()
        smart_selector_mod.return_value = mock_selector
        mock_checker = MagicMock()
        validation_checker_mod.return_value = mock_checker
        mock_df = MagicMock()
        mock_dftypes = MagicMock()
        mock_err_1 = MagicMock(CellErrors)
        mock_err_2 = MagicMock(CellErrors)
        mock_selector.find_actual_errors.return_value = iter([mock_err_1, mock_err_2])

        detector = ValidationDetector()
        detector._df = mock_df
        detector._dftypes = mock_dftypes
        mock_checker.detect_most_possible_candidates = mock_detect

        errors = detector.detect_cells()

        assert mock_detect.called
        assert mock_selector.find_actual_errors.called
        assert mock_selector.find_actual_errors.call_args_list == [
            call(mock_df, mock_dftypes, mock_checker.mentioned_columns)
        ]
        assert errors == CellErrors(MagicMock(), [mock_err_1, mock_err_2])
