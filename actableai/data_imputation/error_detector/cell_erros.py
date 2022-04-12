from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Set

from actableai.data_imputation.meta.column import (
    ColumnName,
    NumWithTagColumnMeta,
)
from actableai.data_imputation.type_recon.type_detector import DfTypes


class ErrorType(Enum):
    NULL = "NULL"
    INVALID = "INVALID"
    MISPLACED = "MISPLACED"
    TYPO = "TYPO"


@dataclass(frozen=True)
class CellInfo:
    column: ColumnName
    index: int


class CellError:
    def __init__(self, column: ColumnName, index: int, error_type: ErrorType):
        self._cell: CellInfo = CellInfo(column, index)
        self._err: ErrorType = error_type

    def __str__(self):
        return f"CellError(column={self._cell.column}, index={self._cell.index}, error={self._err.value})"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if not isinstance(other, CellError):
            return False

        return self._cell == other._cell and self._err == other._err

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(f"{self._cell.index}_{self._cell.column}")

    @property
    def column(self):
        return self._cell.column

    @property
    def index(self):
        return self._cell.index

    @property
    def error_type(self):
        return self._err


class ColumnErrors:
    def __init__(self, column_errors: Set[CellError]):
        assert (
            len(set([err.column for err in column_errors])) < 2
        ), "Errors does not belong to the same column"
        self._errors = column_errors

    def __iter__(self):
        for err in self._errors:
            yield err

    def __eq__(self, other):
        if isinstance(other, ColumnErrors):
            if len(self._errors) != len(other._errors):
                return False
            for x, y in zip(
                sorted(self._errors, key=lambda x: f"{x.column}_{x.index}"),
                sorted(other._errors, key=lambda x: f"{x.column}_{x.index}"),
            ):
                if x != y:
                    return False
            return True
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __len__(self):
        return len(self._errors)

    @property
    def column(self):
        return next(iter(self._errors)).column


class CellErrors:
    def __init__(self, dftypes: DfTypes, errors: List[CellError] = ()):
        self._errors = set(errors)
        self._errors_by_column: Dict[ColumnName, ColumnErrors] = dict()
        self._dftypes: DfTypes = dftypes

    def __len__(self):
        return len(self._errors)

    def __getitem__(self, column: ColumnName) -> ColumnErrors:
        errors = set()
        for err in self._errors:
            if err.column == column:
                errors.add(err)
        return ColumnErrors(errors)

    def __iter__(self) -> CellError:
        for error in self._errors:
            if not self._dftypes.is_support_fix(error.column):
                continue
            col_meta = self._dftypes.get_meta(error.column)
            if isinstance(col_meta, NumWithTagColumnMeta):
                num_col_name = col_meta.get_num_column_name()
                yield CellError(num_col_name, error.index, error.error_type)
                l_tag_col_name = col_meta.get_left_tag_column_name()
                yield CellError(l_tag_col_name, error.index, error.error_type)
                r_tag_col_name = col_meta.get_right_tag_column_name()
                yield CellError(r_tag_col_name, error.index, error.error_type)
            else:
                yield error

    def __str__(self):
        return str(self._errors)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if isinstance(other, CellErrors):
            return self._errors == other._errors
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    @property
    def columns(self) -> Set[ColumnName]:
        return {err.column for err in self}

    def append(self, error: "CellError"):
        self._errors.add(error)

    def extend(self, errors: "CellErrors"):
        for e in errors:
            self.append(e)


@dataclass
class ErrorColumns:
    columns_from_when: Set[ColumnName]
    columns_from_then: Set[ColumnName]


@dataclass
class ErrorCandidate:
    index: int
    columns: ErrorColumns

    @property
    def potential_columns(self) -> Set[ColumnName]:
        return self.columns.columns_from_when.union(self.columns.columns_from_then)
