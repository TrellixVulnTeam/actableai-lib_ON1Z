from collections import defaultdict
from typing import Dict, List

import numpy as np
import pandas as pd

from actableai.data_imputation.meta.column import ColumnName


class CorrelationCalculator:
    def __init__(self):
        self._correlation_scores: Dict[
            ColumnName, Dict[ColumnName, float]
        ] = defaultdict(dict)

    @staticmethod
    def _normalize_column(df: pd.DataFrame, col: ColumnName) -> pd.Series:
        if np.all(pd.isnull(df[col])):
            return pd.Series(dtype="object")

        if df[col].dtype == "object":
            source_ndarray, _ = pd.factorize(df[col])
            source = pd.Series(source_ndarray)
        elif df[col].dtype == "datetime64[ns]":
            source = pd.Series()
        else:
            source = df[col]
        return source

    def _calculate_correlation(self, df: pd.DataFrame, source_column: ColumnName):
        if source_column not in self._correlation_scores:
            for col in df.columns:
                if col == source_column:
                    continue

                source = self._normalize_column(df, source_column)
                target = self._normalize_column(df, col)
                if source.size == 0 or target.size == 0:
                    continue
                self._correlation_scores[source_column][col] = source.corr(target)

        return self._correlation_scores[source_column]

    def most_correlate_columns(
        self, df: pd.DataFrame, col: ColumnName, top: int
    ) -> List[ColumnName]:
        scores = self._calculate_correlation(df, col).items()
        return [k for k, v in sorted(scores, key=lambda x: abs(x[1]), reverse=True)][
            :top
        ]

    def calculate_correlations_for_all_column_pairs(
        self, df: pd.DataFrame
    ) -> pd.DataFrame:
        for col in df.columns:
            self._calculate_correlation(df, col)

        correlations = pd.DataFrame(
            data={col: [0] * len(df.columns) for col in df.columns},
            index=df.columns,
            dtype="float64",
        )

        for col in df.columns:
            correlations.at[col, col] = 1

        for col1 in self._correlation_scores:
            for col2 in self._correlation_scores[col1]:
                correlations.at[col1, col2] = self._correlation_scores[col1][col2]

        return correlations

    def clear(self):
        self._correlation_scores.clear()
