import os.path
import shutil
import time
from enum import Enum
from typing import List, Set, Tuple, Optional

import pandas as pd
from actableai.utils import memory_efficient_hyperparameters
from autogluon.core.dataset import TabularDataset
from autogluon.tabular import TabularPredictor

from actableai.data_imputation.auto_fixer.auto_fixer import AutoFixer
from actableai.data_imputation.auto_fixer.errors import EmptyTrainDataException
from actableai.data_imputation.auto_fixer.fix_info import (
    FixInfoList,
    FixInfo,
    FixValueOptions,
    FixValue,
)
from actableai.data_imputation.auto_fixer.helper import (
    get_df_without_error,
    get_df_with_only_error,
)
from actableai.data_imputation.error_detector import CellErrors
from actableai.data_imputation.meta import ColumnType
from actableai.data_imputation.meta.column import ColumnName, RichColumnMeta


class _ProblemType(Enum):
    binary = "binary"
    multiclass = "multiclass"
    regression = "regression"
    unknown = None


class AutoGluonFixer(AutoFixer):
    def __init__(self):
        """AutoGluonFixer is a fixer that uses AutoGluon to predict missing values"""
        super(AutoGluonFixer, self).__init__()
        self._model_location = _MODEL_LOCATION = f"./AutogluonModels_{time.time()}"

    @staticmethod
    def _decide_problem_type(
        column: RichColumnMeta, target_values: Set
    ) -> _ProblemType:
        """Decides the problem type of the column based on the target values

        Args:
            column: The column to decide the problem type for
            target_values: The target values of the column

        Raises:
            ValueError: If the values only have one unique value

        Returns:
            _ProblemType: The problem type of the column
        """
        if column.type is ColumnType.Category:
            target_values_uniq_count = len(target_values)
            if target_values_uniq_count == 1:
                raise ValueError(
                    "Column values only have 1 unique value, should not use AutoGluonFixer"
                )
            elif len(target_values) == 2:
                return _ProblemType.binary
            else:
                return _ProblemType.multiclass
        elif column.type in [ColumnType.Integer, ColumnType.Float]:
            return _ProblemType.regression

        return _ProblemType.unknown

    def _remove_saved_model(self):
        """Removes the saved model
        """
        if os.path.exists(self._model_location):
            shutil.rmtree(self._model_location, ignore_errors=True)

    def _predict_missing_for_single_column(
        self,
        df: pd.DataFrame,
        columns_to_train: List[ColumnName],
        all_errors: CellErrors,
        column_to_predict: RichColumnMeta,
        hyperparameters: Optional[Tuple[str, dict]] = None,
    ) -> Tuple[_ProblemType, pd.Series]:
        """Predicts the missing values for a single column

        Args:
            df: Input dataframe
            columns_to_train: The columns to train the model on
            all_errors: All errors of the dataframe
            column_to_predict: The column to predict the missing values for

        Raises:
            EmptyTrainDataException: If the dataframe only contains errors

        Returns:
            Tuple[_ProblemType, pd.Series]: 
                - The problem type of the column
                - _description_
        """
        dataset = df[columns_to_train + [column_to_predict.name]]

        if hyperparameters is None:
            hyperparameters = memory_efficient_hyperparameters()

        df_without_error = get_df_without_error(
            dataset, all_errors[column_to_predict.name]
        )

        if df_without_error.empty:
            raise EmptyTrainDataException()

        df_to_train = TabularDataset(df_without_error)
        problem_type = self._decide_problem_type(
            column_to_predict,
            target_values=set(df_to_train[column_to_predict.name]),
        )

        predictor = TabularPredictor(
            label=column_to_predict.name,
            problem_type=problem_type.value,
            path=self._model_location,
        )
        predictor.fit(
            df_to_train,
            hyperparameters=hyperparameters,
            excluded_model_types=["CAT"],
        )
        pd.set_option("chained_assignment", "warn")

        df_to_test = get_df_with_only_error(
            dataset[columns_to_train], all_errors[column_to_predict.name]
        )
        df_to_test = TabularDataset(df_to_test[columns_to_train])
        predict_df_with_confidence = predictor.predict_proba(df_to_test)

        return problem_type, predict_df_with_confidence

    def fix(
        self,
        df: pd.DataFrame,
        all_errors: CellErrors,
        current_column: RichColumnMeta,
        ag_hyperparameters: Optional[Tuple[str, dict]] = None,
    ) -> FixInfoList:
        """Fixes the missing values for a single column

        Args:
            df: Input dataframe
            all_errors: All errors of the dataframe
            current_column: The column to fix the missing values for

        Returns:
            FixInfoList: The fix information for the column
        """
        self._remove_saved_model()

        columns_to_train = set(df.columns)
        columns_to_train.discard(current_column.name)
        problem_type, series_with_fix = self._predict_missing_for_single_column(
            df, list(columns_to_train), all_errors, current_column, ag_hyperparameters
        )
        fix_info_list = FixInfoList()
        for err in all_errors[current_column.name]:
            if problem_type == _ProblemType.regression:
                options = FixValueOptions(
                    options=[
                        FixValue(
                            value=series_with_fix.loc[err.index],
                            confidence=1,
                        )
                    ]
                )
            else:
                options = FixValueOptions(
                    options=[
                        FixValue(
                            value=value,
                            confidence=series_with_fix.at[err.index, value],
                        )
                        for value in series_with_fix.columns
                    ]
                )

            fix_info_list.append(
                FixInfo(
                    col=current_column.name,
                    index=err.index,
                    options=options,
                )
            )

        self._remove_saved_model()

        return fix_info_list
