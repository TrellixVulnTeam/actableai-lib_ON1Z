import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from math import inf
from typing import Tuple, List

from actableai.data_imputation.auto_fixer.auto_gluon_fixer import AutoGluonFixer
from actableai.data_imputation.auto_fixer.fix_info import (
    FixInfoList,
    FixInfo,
    FixValue,
    FixValueOptions,
)
from actableai.data_imputation.auto_fixer.single_category_fixer import (
    SingleCategoryFixer,
)
from actableai.data_imputation.auto_fixer.strategy import (
    determine_refine_strategy,
    FixStrategy,
)
from actableai.data_imputation.config import logger, UNABLE_TO_FIX_PLACEHOLDER
from actableai.data_imputation.correlation_calculator import (
    CorrelationCalculator,
)
from actableai.data_imputation.error_detector import ErrorDetector
from actableai.data_imputation.error_detector.cell_erros import (
    CellErrors,
)
from actableai.data_imputation.error_detector.constraint import Constraints
from actableai.data_imputation.error_detector.misplaced_detector import (
    MisplacedDetector,
)
from actableai.data_imputation.error_detector.validation_constraints_checker import (
    ValidationConstrainsChecker,
)
from actableai.data_imputation.meta.column import RichColumnMeta
from actableai.data_imputation.type_recon.type_detector import (
    ColumnType,
    DfTypes,
)


class Refiner(ABC):
    def __init__(self, error_detector: ErrorDetector):
        pass

    @staticmethod
    def _find_all_possible_fixes(
        errors: CellErrors,
        processed_fixed_df: pd.DataFrame,
    ) -> FixInfoList:
        all_fix_info_list = FixInfoList()
        for i, col in enumerate(errors.columns):
            logger.info(
                f"Finding all possible fixes for column {col} - {i+1}/{len(errors.columns)}"
            )
            strategy = determine_refine_strategy(processed_fixed_df[col], errors[col])
            if strategy == FixStrategy.SINGLE_CATEGORY:
                fixer = SingleCategoryFixer()
            elif strategy == FixStrategy.AUTOGLUON:
                fixer = AutoGluonFixer()
            else:
                raise NotImplementedError

            rich_column_meta_dict = {
                "float64": ColumnType.Float,
                "category": ColumnType.Category,
                "object": ColumnType.Category,
                "int64": ColumnType.Integer,
            }

            fix_info_list = fixer.fix(
                processed_fixed_df,
                errors,
                RichColumnMeta(
                    col, rich_column_meta_dict[str(processed_fixed_df[col].dtype)]
                ),
            )
            all_fix_info_list.extend(fix_info_list)
        return all_fix_info_list

    @staticmethod
    def _get_best_fix_pair(
        fix_info_list: FixInfoList,
        row_index: int,
        correlations: pd.DataFrame,
        possible_value_pairs: pd.DataFrame,
    ) -> List[FixInfo]:
        """

        : params
          fix_info_for_row: list of fix info, should not have multiple item with same index and columns
        """
        columns = fix_info_list.columns

        best = [-inf, pd.Series(dtype="object")]
        for index, row in possible_value_pairs.iterrows():
            selected_columns = row[columns]
            score = 0
            for column in selected_columns.index:
                value = selected_columns[column]
                confidence = fix_info_list.find(row_index, column, value).confidence
                correlations_for_column = correlations[column]
                score += np.sum(confidence * correlations_for_column)

            if score > best[0]:
                best[0] = score
                best[1] = row

        if best[0] == -inf:
            return []

        best_pair = best[1]
        return [
            FixInfo(
                col=column,
                index=row_index,
                options=FixValueOptions(
                    options=[fix_info_list.find(row_index, column, best_pair[column])]
                ),
            )
            for column in best_pair.index
            if column in columns
        ]

    @staticmethod
    def _replace_as_unable_to_fix_in_place(
        errors: CellErrors, df: pd.DataFrame, fix_info_list: FixInfoList
    ):
        for err in errors:
            df.loc[err.index, err.column] = UNABLE_TO_FIX_PLACEHOLDER
            fix_info_list.replace(
                FixInfo(
                    col=err.column,
                    index=err.index,
                    options=FixValueOptions(
                        options=[
                            FixValue(value=UNABLE_TO_FIX_PLACEHOLDER, confidence=-inf)
                        ]
                    ),
                )
            )

    def _replace_best_fix_in_place(
        self, errors: CellErrors, df: pd.DataFrame, fix_info_list: FixInfoList
    ):
        pairs = self._find_all_value_pairs_satisfy_constraints(errors, df)
        if pairs.empty:
            return

        correlation_calculator = CorrelationCalculator()
        correlation_between_columns = (
            correlation_calculator.calculate_correlations_for_all_column_pairs(df)
        )

        for index in fix_info_list.index:
            fix_info_for_row = list(fix_info_list.get_via_index(index))
            best_fix = self._get_best_fix_pair(
                FixInfoList(fix_info_for_row),
                index,
                correlation_between_columns,
                pairs,
            )
            fix_info_list.replace(*best_fix)

        for info in fix_info_list:
            df.loc[info.index, info.col] = info.best_guess

    @abstractmethod
    def _find_all_value_pairs_satisfy_constraints(
        self, errors: CellErrors, df: pd.DataFrame
    ) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def _detect_not_satisfied_cells(
        self,
        df: pd.DataFrame,
    ) -> CellErrors:
        raise NotImplementedError

    @abstractmethod
    def refine(
        self,
        processed_fixed_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, FixInfoList]:
        pass
