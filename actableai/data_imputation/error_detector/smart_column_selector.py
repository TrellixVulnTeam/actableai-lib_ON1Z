from math import inf
from typing import List, Iterable, Tuple, Set

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_distances

from actableai.data_imputation.config import logger
from actableai.data_imputation.correlation_calculator import (
    CorrelationCalculator,
)
from actableai.data_imputation.error_detector.cell_erros import (
    ErrorCandidate,
    CellError,
    ErrorType,
)
from actableai.data_imputation.meta.column import ColumnName, ColumnType
from actableai.data_imputation.type_recon.type_detector import DfTypes


class SmartErrorCellSelector:
    def __init__(self):
        self._invalid_candidates: List[ErrorCandidate] = []
        self._correlation_calculator = CorrelationCalculator()

    def reset(self):
        self._invalid_candidates.clear()
        self._correlation_calculator.clear()

    def append_candidate(self, candidate: ErrorCandidate):
        self._invalid_candidates.append(candidate)

    @staticmethod
    def _distance_to_center(
        df_original: pd.DataFrame,
        dftypes: DfTypes,
        column: ColumnName,
        index: int,
    ) -> float:
        df = df_original.copy()
        for col in df.columns:
            if dftypes[col] not in [
                ColumnType.Integer,
                ColumnType.Float,
            ]:
                df[col] = pd.factorize(df[col])[0]

        value = df.at[index, column]
        row = df.loc[index].to_numpy()

        df_trimmed = df[df[column] == value].drop(index, axis=0)
        df_trimmed.dropna(inplace=True)
        df_trimmed.reset_index(drop=True, inplace=True)

        if df_trimmed.size == 0:
            return inf
        kmean = KMeans(n_clusters=1)
        kmean.fit(df_trimmed)
        center = kmean.cluster_centers_[0]

        nan_indexes = np.argwhere(pd.isna(row))
        if len(nan_indexes) > 0:
            row[nan_indexes] = center[nan_indexes]

        distance = cosine_distances([row], [center])[0]
        return distance[0]

    @staticmethod
    def _find_columns_with_uniq_value(df: pd.DataFrame, index: int) -> Set[ColumnName]:
        if df.empty:
            return set()

        columns = set()
        for col in df.columns:
            val = df.at[index, col]
            value_counts = df[col].value_counts()
            if value_counts[val] == 1:
                columns.add(col)
        return columns

    def _find_actual_error_column(
        self,
        df: pd.DataFrame,
        dftypes: DfTypes,
        candidate: ErrorCandidate,
        focused_columns: Set[ColumnName],
    ) -> Iterable[CellError]:
        potential_columns = candidate.potential_columns
        non_nan_potential_columns = []

        for col in potential_columns:
            if pd.isna(df.at[candidate.index, col]):
                yield CellError(
                    column=col,
                    index=candidate.index,
                    error_type=ErrorType.INVALID,
                )
            else:
                non_nan_potential_columns.append(col)

        columns_with_uniq_value = self._find_columns_with_uniq_value(
            df[non_nan_potential_columns], candidate.index
        )

        distance_column_pair = {}  # column to distance pair
        for col in sorted(non_nan_potential_columns):
            most_correlate_columns = (
                self._correlation_calculator.most_correlate_columns(df, col, top=5)
            )
            columns = set(most_correlate_columns).intersection(set(focused_columns))
            columns = columns.difference(columns_with_uniq_value)

            if len(columns) == 0:
                yield CellError(
                    column=col,
                    index=candidate.index,
                    error_type=ErrorType.INVALID,
                )
            else:
                columns.add(col)
                distance = self._distance_to_center(
                    df[columns],
                    dftypes,
                    col,
                    candidate.index,
                )
                distance_column_pair[col] = distance

        if not distance_column_pair:
            return

        distances: List[Tuple[str, float]] = sorted(
            [(k, distance_column_pair[k]) for k in distance_column_pair],
            key=lambda x: x[1],
        )
        inf_list = []
        non_inf_list = []
        for col, distance in distances:
            if distance == inf:
                inf_list.append((col, distance))
            else:
                non_inf_list.append((col, distance))

        for col, distance in inf_list:
            yield CellError(
                column=col,
                index=candidate.index,
                error_type=ErrorType.INVALID,
            )

        non_inf_cols = [v[0] for v in non_inf_list]
        non_inf_distances = np.asarray([v[1] for v in non_inf_list])
        if len(set(non_inf_distances)) <= 1:
            return
        non_inf_distances = non_inf_distances / np.linalg.norm([non_inf_distances])
        max_distance = np.max(non_inf_distances)
        for col, distance in zip(non_inf_cols, non_inf_distances):
            if max_distance - distance < 0.1:
                yield CellError(
                    column=col,
                    index=candidate.index,
                    error_type=ErrorType.INVALID,
                )

    def find_actual_errors(
        self,
        df: pd.DataFrame,
        dftypes: DfTypes,
        focused_columns: Set[ColumnName],
    ) -> Iterable[CellError]:
        logger.info(
            f"Finding and expanding actual errors among {len(self._invalid_candidates)} candidates"
        )
        errors = []
        for candidate in self._invalid_candidates:
            for err in self._find_actual_error_column(
                df, dftypes, candidate, focused_columns
            ):
                errors.append(err)
                yield err
