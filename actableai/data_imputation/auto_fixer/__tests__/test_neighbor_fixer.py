from unittest.mock import patch, MagicMock

import pandas as pd
import pytest
from pandas import Index
from actableai.data_imputation.auto_fixer.errors import EmptyTrainDataException
from actableai.data_imputation.auto_fixer.neighbor_fixer import NeighborFixer
from actableai.data_imputation.error_detector import CellErrors
from actableai.data_imputation.error_detector.cell_erros import CellError
from actableai.data_imputation.type_recon.type_detector import DfTypes

stub_path = "actableai.data_imputation.auto_fixer.neighbor_fixer"


@pytest.mark.parametrize("df_without_error", [pd.DataFrame.from_dict({"A": [1, 2]})])
@patch(f"{stub_path}.get_df_without_error")
@patch(f"{stub_path}.IterativeImputer")
def test_fix_should_raise_error_when_df_without_error_is_empty(
    mock_interactive_imputer,
    mock_get_df_without_error,
    df_without_error,
):
    df = MagicMock()
    errors = MagicMock()
    current_column = MagicMock()
    mock_get_df_without_error.return_value = pd.DataFrame()

    fixer = NeighborFixer()
    with pytest.raises(EmptyTrainDataException):
        fixer.fix(df, errors, current_column)


@pytest.mark.parametrize("df_without_error", [pd.DataFrame.from_dict({"A": [1, 2]})])
@patch(f"{stub_path}.get_df_without_error")
@patch(f"{stub_path}.IterativeImputer")
def test_fix(
    mock_interactive_imputer,
    mock_get_df_without_error,
    df_without_error,
):
    df = MagicMock()
    errors = [MagicMock(CellError), MagicMock(CellError), MagicMock(CellError)]
    all_errors = MagicMock(CellErrors)
    all_errors.__getitem__.return_value = errors
    df_fixed_on_error = MagicMock()
    current_column_rich_meta = MagicMock()
    current_column = MagicMock()
    current_column_rich_meta.name = current_column

    df.columns = [current_column]
    df.columns = Index([current_column])
    imp = MagicMock()
    mock_interactive_imputer.return_value = imp
    imp.fit_transform.return_value = df_fixed_on_error

    mock_get_df_without_error.return_value = df_without_error

    fixer = NeighborFixer()
    results = fixer.fix(df, all_errors, current_column_rich_meta)

    assert imp.fit_transform.call_count == 1
    assert len(results) == len(errors)
