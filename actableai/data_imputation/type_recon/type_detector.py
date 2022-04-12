from typing import Tuple, List, Dict, Set

import pandas as pd

from actableai.data_imputation.meta.column import (
    ColumnName,
    RichColumnMeta,
    SingleValueColumnMeta,
    NumWithTagColumnMeta,
)
from actableai.data_imputation.meta.types import ColumnType
from actableai.data_imputation.type_recon.dtype_detector import (
    build_dtype_detector,
    detect_possible_type_for_column,
)


class DfTypes:
    def __init__(self, results: List[Tuple[ColumnName, ColumnType]]):
        self.__column_names = {r[0] for r in results}
        self.__results: Dict[ColumnName, RichColumnMeta] = {
            r[0]: self._enrich_type(r[0], r[1]) for r in results
        }
        self.__column_unsupported: Set[ColumnName] = set()

    def __getitem__(self, column: ColumnName) -> ColumnType:
        for meta in self.__results.values():
            if isinstance(meta, SingleValueColumnMeta):
                if meta.name == column:
                    return meta.type
            if isinstance(meta, NumWithTagColumnMeta):
                if meta.name == column:
                    return meta.original_type
                if meta.get_num_column_name() == column:
                    return meta.type
                if meta.get_left_tag_column_name() == column:
                    return ColumnType.Category
                if meta.get_right_tag_column_name() == column:
                    return ColumnType.Category

        raise KeyError(f"{column} does not exist in DfTypes")

    def __eq__(self, other):
        return self.__results == other.__results

    def __str__(self):
        return f"{[f'{col}: {self[col].value}' for col in self.__results]}"

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def _enrich_type(col: ColumnName, col_type: ColumnType):
        if col_type in [
            ColumnType.Percentage,
            ColumnType.NumWithTag,
            ColumnType.Temperature,
        ]:
            return NumWithTagColumnMeta(col, col_type)
        else:
            return SingleValueColumnMeta(col, col_type)

    @property
    def columns_original(self):
        # exclude columns unable to fix
        for col in self.__results.keys():
            if self.is_support_fix(col):
                yield col

    @property
    def _columns_after_expand(self):
        # does not care if the column is unsupported
        for meta in self.__results.values():
            if isinstance(meta, SingleValueColumnMeta):
                yield meta.name
            if isinstance(meta, NumWithTagColumnMeta):
                yield meta.get_left_tag_column_name()
                yield meta.get_num_column_name()
                yield meta.get_right_tag_column_name()

    @property
    def _columns_all(self):
        # does not care if the column is unsupported
        for meta in self.__results.values():
            if isinstance(meta, SingleValueColumnMeta):
                yield meta.name
            if isinstance(meta, NumWithTagColumnMeta):
                yield meta.name
                yield meta.get_left_tag_column_name()
                yield meta.get_num_column_name()
                yield meta.get_right_tag_column_name()

    @property
    def columns_to_fix(self):
        for col in self._columns_after_expand:
            if self.is_support_fix(col):
                yield col

    def get_meta(self, column: ColumnName) -> RichColumnMeta:
        for meta in self.__results.values():
            if isinstance(meta, SingleValueColumnMeta):
                if meta.name == column:
                    return meta
            if isinstance(meta, NumWithTagColumnMeta):
                if meta.name == column:
                    return meta
                if meta.get_num_column_name() == column:
                    return SingleValueColumnMeta(column, meta.type)
                if meta.get_left_tag_column_name() == column:
                    return SingleValueColumnMeta(column, ColumnType.String)
                if meta.get_right_tag_column_name() == column:
                    return SingleValueColumnMeta(column, ColumnType.String)
        raise KeyError(f"Cannot find meta for {column}")

    def mark_column_unsupported(self, col: ColumnName):
        meta = self.get_meta(col)
        if isinstance(meta, SingleValueColumnMeta):
            self.__column_unsupported.add(col)
        elif isinstance(meta, NumWithTagColumnMeta):
            self.__column_unsupported.add(meta.name)
            self.__column_unsupported.add(meta.get_num_column_name())
            self.__column_unsupported.add(meta.get_left_tag_column_name())
            self.__column_unsupported.add(meta.get_right_tag_column_name())
        else:
            raise NotImplementedError

    def override(self, col: ColumnName, col_type: ColumnType):
        if col not in self.__column_names:
            raise KeyError(f"{col} does not appear in the original dataframe")
        self.__results[col] = self._enrich_type(col, col_type)

    def is_support_fix(self, col: ColumnName) -> bool:
        return col not in self.__column_unsupported and col in self._columns_all


class TypeDetector:
    @staticmethod
    def detect_column(series: pd.Series) -> ColumnType:
        detector = build_dtype_detector(series)
        return detector.get_column_type()

    @staticmethod
    def detect_possible_types(
        df: pd.DataFrame,
    ) -> Dict[ColumnName, Set[ColumnType]]:
        return {col: detect_possible_type_for_column(df[col]) for col in df.columns}

    def detect(self, df: pd.DataFrame) -> DfTypes:
        return DfTypes([(col, self.detect_column(df[col])) for col in df.columns])
