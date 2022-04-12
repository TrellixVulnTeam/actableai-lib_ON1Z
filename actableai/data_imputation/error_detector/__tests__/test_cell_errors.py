import time

import pytest

from actableai.data_imputation.error_detector.cell_erros import (
    CellErrors,
    CellError,
    ErrorType,
    ColumnErrors,
)
from actableai.data_imputation.meta.types import ColumnType
from actableai.data_imputation.type_recon.type_detector import DfTypes


def test_cell_errors_everytime_initial_should_be_different():
    e1 = CellErrors(DfTypes([("a", ColumnType.Integer)]))
    e1.append(CellError("a", 1, ErrorType.INVALID))

    e2 = CellErrors(DfTypes([("a", ColumnType.Integer)]))
    e2.append(CellError("a", 2, ErrorType.INVALID))

    assert len(e1) == 1
    assert len(e2) == 1

    e2.append(CellError("a", 3, ErrorType.INVALID))
    assert len(e1) == 1
    assert len(e2) == 2


def test_cell_errors_append_should_deduplicate():
    e = CellErrors(DfTypes([("a", ColumnType.Integer)]))
    e.append(CellError("a", 1, ErrorType.INVALID))
    e.append(CellError("a", 2, ErrorType.INVALID))
    e.append(CellError("a", 1, ErrorType.INVALID))

    assert len(e) == 2


def test_cell_errors_extend_should_deduplicate():
    e1 = CellErrors(DfTypes([("a", ColumnType.Integer)]))
    e1.append(CellError("a", 1, ErrorType.INVALID))
    e1.append(CellError("a", 2, ErrorType.INVALID))

    e2 = CellErrors(DfTypes([("a", ColumnType.Integer)]))
    e2.append(CellError("a", 1, ErrorType.INVALID))

    e1.extend(e2)

    assert len(e1) == 2


def test_cell_errors_columns():
    e1 = CellErrors(DfTypes([("a", ColumnType.Integer), ("b", ColumnType.Integer)]))
    e1.append(CellError("a", 7, ErrorType.INVALID))
    e1.append(CellError("a", 3, ErrorType.INVALID))
    e1.append(CellError("b", 9, ErrorType.INVALID))

    assert e1.columns == {"a", "b"}


def test_cell_errors_columns_should_ignore_unsupported_columns():
    dftypes = DfTypes([("a", ColumnType.Integer), ("b", ColumnType.Integer)])
    e1 = CellErrors(dftypes)
    e1.append(CellError("a", 7, ErrorType.INVALID))
    e1.append(CellError("a", 3, ErrorType.INVALID))
    e1.append(CellError("b", 9, ErrorType.INVALID))
    dftypes.mark_column_unsupported("a")

    assert e1.columns == {"b"}


def test_cell_errors_append_large_list():
    start = time.time()
    e1 = CellErrors(DfTypes([("a", ColumnType.Integer)]))
    size = 1_000_000
    for i in range(size):
        e1.append(CellError("a", i, ErrorType.NULL))
    assert len(e1) == size
    end = time.time()
    print("Total time cost:", end - start)


@pytest.mark.parametrize(
    "cell_errors, expect_errors",
    [
        (
            CellErrors(
                DfTypes([("a", ColumnType.Integer)]),
                [CellError("a", 7, ErrorType.INVALID)],
            ),
            [CellError("a", 7, ErrorType.INVALID)],
        ),
        (
            CellErrors(
                DfTypes([("a", ColumnType.Percentage)]),
                [CellError("a", 7, ErrorType.INVALID)],
            ),
            [
                CellError("__a_num__", 7, ErrorType.INVALID),
                CellError("__a_ltag__", 7, ErrorType.INVALID),
                CellError("__a_rtag__", 7, ErrorType.INVALID),
            ],
        ),
    ],
)
def test_cell_errors_iter(cell_errors, expect_errors):
    actual_errors = list(cell_errors)
    assert actual_errors == expect_errors


def test_cell_errors_iter_should_ignore_unsupported_columns():
    dftypes = DfTypes([("a", ColumnType.Integer)])
    dftypes.mark_column_unsupported("a")
    errors = CellErrors(dftypes, [CellError("a", 7, ErrorType.INVALID)])
    assert list(errors) == []


@pytest.mark.parametrize(
    "errors, column",
    [
        (
            ColumnErrors(
                {
                    CellError("a", 7, ErrorType.INVALID),
                    CellError("a", 8, ErrorType.INVALID),
                }
            ),
            "a",
        )
    ],
)
def test_column_errors_get_column(errors, column):
    assert errors.column == column


@pytest.mark.parametrize(
    "errors, expect_size",
    [
        (
            ColumnErrors(
                {
                    CellError("a", 7, ErrorType.INVALID),
                    CellError("a", 8, ErrorType.INVALID),
                }
            ),
            2,
        )
    ],
)
def test_column_errors_get_column_does_not_pop_item_from_errors(errors, expect_size):
    _ = errors.column
    assert len(errors) == expect_size
