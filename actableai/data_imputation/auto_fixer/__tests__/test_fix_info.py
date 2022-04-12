import time
from math import inf

import pytest

from actableai.data_imputation.auto_fixer.__tests__.helper import (
    assert_fix_info_list,
)
from actableai.data_imputation.auto_fixer.fix_info import (
    FixInfo,
    FixInfoList,
    FixValueOptions,
    FixValue,
)
from actableai.data_imputation.config import UNABLE_TO_FIX_PLACEHOLDER


@pytest.mark.parametrize(
    "fix_info_list, expect_fix_info_list",
    [
        (
            [
                FixInfo(
                    col="A",
                    index=1,
                    options=FixValueOptions(
                        options=[FixValue(value="a", confidence=1)]
                    ),
                ),
                FixInfo(
                    col="A",
                    index=2,
                    options=FixValueOptions(
                        options=[FixValue(value="a", confidence=1)]
                    ),
                ),
                FixInfo(
                    col="A",
                    index=3,
                    options=FixValueOptions(
                        options=[FixValue(value="a", confidence=1)]
                    ),
                ),
                FixInfo(
                    col="A",
                    index=4,
                    options=FixValueOptions(
                        options=[FixValue(value="a", confidence=1)]
                    ),
                ),
            ],
            [
                FixInfo(
                    col="A",
                    index=1,
                    options=FixValueOptions(
                        options=[FixValue(value="a", confidence=1)]
                    ),
                ),
                FixInfo(
                    col="A",
                    index=2,
                    options=FixValueOptions(
                        options=[FixValue(value="a", confidence=1)]
                    ),
                ),
                FixInfo(
                    col="A",
                    index=3,
                    options=FixValueOptions(
                        options=[FixValue(value="a", confidence=1)]
                    ),
                ),
                FixInfo(
                    col="A",
                    index=4,
                    options=FixValueOptions(
                        options=[FixValue(value="a", confidence=1)]
                    ),
                ),
            ],
        ),
        (
            [
                FixInfo(
                    col="A",
                    index=1,
                    options=FixValueOptions(
                        options=[FixValue(value="a", confidence=1)]
                    ),
                ),
                FixInfo(
                    col="A",
                    index=1,
                    options=FixValueOptions(
                        options=[FixValue(value="a", confidence=1)]
                    ),
                ),
                FixInfo(
                    col="A",
                    index=1,
                    options=FixValueOptions(
                        options=[FixValue(value="a", confidence=1)]
                    ),
                ),
                FixInfo(
                    col="A",
                    index=4,
                    options=FixValueOptions(
                        options=[FixValue(value="a", confidence=1)]
                    ),
                ),
            ],
            [
                FixInfo(
                    col="A",
                    index=1,
                    options=FixValueOptions(
                        options=[FixValue(value="a", confidence=1)]
                    ),
                ),
                FixInfo(
                    col="A",
                    index=4,
                    options=FixValueOptions(
                        options=[FixValue(value="a", confidence=1)]
                    ),
                ),
            ],
        ),
        (
            [
                FixInfo(
                    col="A",
                    index=1,
                    options=FixValueOptions(
                        options=[FixValue(value="a", confidence=1)]
                    ),
                ),
                FixInfo(
                    col="A",
                    index=1,
                    options=FixValueOptions(
                        options=[FixValue(value="a", confidence=1)]
                    ),
                ),
                FixInfo(
                    col="A",
                    index=1,
                    options=FixValueOptions(
                        options=[FixValue(value="c", confidence=1)]
                    ),
                ),
                FixInfo(
                    col="A",
                    index=4,
                    options=FixValueOptions(
                        options=[FixValue(value="a", confidence=1)]
                    ),
                ),
            ],
            [
                FixInfo(
                    col="A",
                    index=1,
                    options=FixValueOptions(
                        options=[FixValue(value="c", confidence=1)]
                    ),
                ),
                FixInfo(
                    col="A",
                    index=4,
                    options=FixValueOptions(
                        options=[FixValue(value="a", confidence=1)]
                    ),
                ),
            ],
        ),
    ],
)
def test_fix_info_list_init(fix_info_list, expect_fix_info_list):
    assert_fix_info_list(FixInfoList(fix_info_list), expect_fix_info_list)


@pytest.mark.parametrize(
    "fix_info_list, expect_fix_info_list",
    [
        (
            [
                FixInfo(
                    col="A",
                    index=1,
                    options=FixValueOptions(
                        options=[FixValue(value="a", confidence=1)]
                    ),
                ),
                FixInfo(
                    col="A",
                    index=2,
                    options=FixValueOptions(
                        options=[FixValue(value="a", confidence=1)]
                    ),
                ),
                FixInfo(
                    col="A",
                    index=3,
                    options=FixValueOptions(
                        options=[FixValue(value="a", confidence=1)]
                    ),
                ),
                FixInfo(
                    col="A",
                    index=4,
                    options=FixValueOptions(
                        options=[FixValue(value="a", confidence=1)]
                    ),
                ),
            ],
            [
                FixInfo(
                    col="A",
                    index=1,
                    options=FixValueOptions(
                        options=[FixValue(value="a", confidence=1)]
                    ),
                ),
                FixInfo(
                    col="A",
                    index=2,
                    options=FixValueOptions(
                        options=[FixValue(value="a", confidence=1)]
                    ),
                ),
                FixInfo(
                    col="A",
                    index=3,
                    options=FixValueOptions(
                        options=[FixValue(value="a", confidence=1)]
                    ),
                ),
                FixInfo(
                    col="A",
                    index=4,
                    options=FixValueOptions(
                        options=[FixValue(value="a", confidence=1)]
                    ),
                ),
            ],
        ),
        (
            [
                FixInfo(
                    col="A",
                    index=1,
                    options=FixValueOptions(
                        options=[FixValue(value="a", confidence=1)]
                    ),
                ),
                FixInfo(
                    col="A",
                    index=1,
                    options=FixValueOptions(
                        options=[FixValue(value="a", confidence=1)]
                    ),
                ),
                FixInfo(
                    col="A",
                    index=1,
                    options=FixValueOptions(
                        options=[FixValue(value="a", confidence=1)]
                    ),
                ),
                FixInfo(
                    col="A",
                    index=4,
                    options=FixValueOptions(
                        options=[FixValue(value="a", confidence=1)]
                    ),
                ),
            ],
            [
                FixInfo(
                    col="A",
                    index=1,
                    options=FixValueOptions(
                        options=[FixValue(value="a", confidence=1)]
                    ),
                ),
                FixInfo(
                    col="A",
                    index=4,
                    options=FixValueOptions(
                        options=[FixValue(value="a", confidence=1)]
                    ),
                ),
            ],
        ),
        (
            [
                FixInfo(
                    col="A",
                    index=1,
                    options=FixValueOptions(
                        options=[FixValue(value="a", confidence=1)]
                    ),
                ),
                FixInfo(
                    col="A",
                    index=1,
                    options=FixValueOptions(
                        options=[FixValue(value="a", confidence=1)]
                    ),
                ),
                FixInfo(
                    col="A",
                    index=1,
                    options=FixValueOptions(
                        options=[FixValue(value="c", confidence=1)]
                    ),
                ),
                FixInfo(
                    col="A",
                    index=4,
                    options=FixValueOptions(
                        options=[FixValue(value="a", confidence=1)]
                    ),
                ),
            ],
            [
                FixInfo(
                    col="A",
                    index=1,
                    options=FixValueOptions(
                        options=[FixValue(value="c", confidence=1)]
                    ),
                ),
                FixInfo(
                    col="A",
                    index=4,
                    options=FixValueOptions(
                        options=[FixValue(value="a", confidence=1)]
                    ),
                ),
            ],
        ),
    ],
)
def test_fix_info_list_append(fix_info_list, expect_fix_info_list):
    actual_list = FixInfoList()
    for v in fix_info_list:
        actual_list.append(v)
    assert_fix_info_list(actual_list, expect_fix_info_list)


@pytest.mark.parametrize(
    "fix_info_list_1, fix_info_list_2, expect_fix_info_list",
    [
        (
            [
                FixInfo(
                    col="A",
                    index=1,
                    options=FixValueOptions(
                        options=[FixValue(value="a", confidence=1)]
                    ),
                ),
                FixInfo(
                    col="A",
                    index=2,
                    options=FixValueOptions(
                        options=[FixValue(value="a", confidence=1)]
                    ),
                ),
            ],
            [
                FixInfo(
                    col="A",
                    index=3,
                    options=FixValueOptions(
                        options=[FixValue(value="a", confidence=1)]
                    ),
                ),
                FixInfo(
                    col="A",
                    index=4,
                    options=FixValueOptions(
                        options=[FixValue(value="a", confidence=1)]
                    ),
                ),
            ],
            [
                FixInfo(
                    col="A",
                    index=1,
                    options=FixValueOptions(
                        options=[FixValue(value="a", confidence=1)]
                    ),
                ),
                FixInfo(
                    col="A",
                    index=2,
                    options=FixValueOptions(
                        options=[FixValue(value="a", confidence=1)]
                    ),
                ),
                FixInfo(
                    col="A",
                    index=3,
                    options=FixValueOptions(
                        options=[FixValue(value="a", confidence=1)]
                    ),
                ),
                FixInfo(
                    col="A",
                    index=4,
                    options=FixValueOptions(
                        options=[FixValue(value="a", confidence=1)]
                    ),
                ),
            ],
        ),
        (
            [
                FixInfo(
                    col="A",
                    index=1,
                    options=FixValueOptions(
                        options=[FixValue(value="a", confidence=1)]
                    ),
                ),
                FixInfo(
                    col="A",
                    index=1,
                    options=FixValueOptions(
                        options=[FixValue(value="a", confidence=1)]
                    ),
                ),
            ],
            [
                FixInfo(
                    col="A",
                    index=1,
                    options=FixValueOptions(
                        options=[FixValue(value="a", confidence=1)]
                    ),
                ),
                FixInfo(
                    col="A",
                    index=4,
                    options=FixValueOptions(
                        options=[FixValue(value="a", confidence=1)]
                    ),
                ),
            ],
            [
                FixInfo(
                    col="A",
                    index=1,
                    options=FixValueOptions(
                        options=[FixValue(value="a", confidence=1)]
                    ),
                ),
                FixInfo(
                    col="A",
                    index=4,
                    options=FixValueOptions(
                        options=[FixValue(value="a", confidence=1)]
                    ),
                ),
            ],
        ),
        (
            [
                FixInfo(
                    col="A",
                    index=1,
                    options=FixValueOptions(
                        options=[FixValue(value="a", confidence=1)]
                    ),
                ),
                FixInfo(
                    col="A",
                    index=1,
                    options=FixValueOptions(
                        options=[FixValue(value="a", confidence=1)]
                    ),
                ),
            ],
            [
                FixInfo(
                    col="A",
                    index=1,
                    options=FixValueOptions(
                        options=[FixValue(value="c", confidence=1)]
                    ),
                ),
                FixInfo(
                    col="A",
                    index=4,
                    options=FixValueOptions(
                        options=[FixValue(value="a", confidence=1)]
                    ),
                ),
            ],
            [
                FixInfo(
                    col="A",
                    index=1,
                    options=FixValueOptions(
                        options=[FixValue(value="c", confidence=1)]
                    ),
                ),
                FixInfo(
                    col="A",
                    index=4,
                    options=FixValueOptions(
                        options=[FixValue(value="a", confidence=1)]
                    ),
                ),
            ],
        ),
    ],
)
def test_fix_info_list_extend(fix_info_list_1, fix_info_list_2, expect_fix_info_list):
    actual_list = FixInfoList()
    actual_list.extend(FixInfoList(fix_info_list_1))
    actual_list.extend(FixInfoList(fix_info_list_2))

    assert_fix_info_list(actual_list, expect_fix_info_list)


@pytest.mark.parametrize(
    "fix_info_list, index, column, expect_fix_info",
    [
        (
            FixInfoList(
                [
                    FixInfo(
                        col="A",
                        index=1,
                        options=FixValueOptions(
                            options=[FixValue(value="a", confidence=1)]
                        ),
                    ),
                    FixInfo(
                        col="A",
                        index=2,
                        options=FixValueOptions(
                            options=[FixValue(value="a", confidence=1)]
                        ),
                    ),
                ]
            ),
            1,
            "A",
            FixInfo(
                col="A",
                index=1,
                options=FixValueOptions(options=[FixValue(value="a", confidence=1)]),
            ),
        ),
        (
            FixInfoList(
                [
                    FixInfo(
                        col="A",
                        index=1,
                        options=FixValueOptions(
                            options=[FixValue(value="a", confidence=1)]
                        ),
                    ),
                    FixInfo(
                        col="A",
                        index=2,
                        options=FixValueOptions(
                            options=[FixValue(value="a", confidence=1)]
                        ),
                    ),
                ]
            ),
            1,
            "B",
            None,
        ),
    ],
)
def test_fix_info_list_get_item(fix_info_list, index, column, expect_fix_info):
    assert fix_info_list.get_item(index, column) == expect_fix_info


@pytest.mark.parametrize(
    "fix_info_list, index, expect_fix_info_list",
    [
        (
            FixInfoList(
                [
                    FixInfo(
                        col="A",
                        index=1,
                        options=FixValueOptions(
                            options=[FixValue(value="a", confidence=1)]
                        ),
                    ),
                    FixInfo(
                        col="A",
                        index=2,
                        options=FixValueOptions(
                            options=[FixValue(value="a", confidence=1)]
                        ),
                    ),
                ]
            ),
            1,
            [
                FixInfo(
                    col="A",
                    index=1,
                    options=FixValueOptions(
                        options=[FixValue(value="a", confidence=1)]
                    ),
                )
            ],
        ),
        (
            FixInfoList(
                [
                    FixInfo(
                        col="A",
                        index=1,
                        options=FixValueOptions(
                            options=[FixValue(value="a", confidence=1)]
                        ),
                    ),
                    FixInfo(
                        col="A",
                        index=2,
                        options=FixValueOptions(
                            options=[FixValue(value="a", confidence=1)]
                        ),
                    ),
                ]
            ),
            3,
            [],
        ),
    ],
)
def test_fix_info_list_get_via_index(fix_info_list, index, expect_fix_info_list):
    assert list(fix_info_list.get_via_index(index)) == expect_fix_info_list


@pytest.mark.parametrize(
    "fix_info_list, index, column, value, expect",
    [
        (
            FixInfoList(
                [
                    FixInfo(
                        col="A",
                        index=1,
                        options=FixValueOptions(
                            options=[
                                FixValue(value="a", confidence=8),
                                FixValue(value="b", confidence=0.2),
                            ]
                        ),
                    ),
                    FixInfo(
                        col="A",
                        index=2,
                        options=FixValueOptions(
                            options=[FixValue(value="a", confidence=1)]
                        ),
                    ),
                ]
            ),
            1,
            "A",
            "b",
            FixValue(value="b", confidence=0.2),
        ),
        (
            FixInfoList(
                [
                    FixInfo(
                        col="A",
                        index=1,
                        options=FixValueOptions(
                            options=[
                                FixValue(value="a", confidence=8),
                                FixValue(value="b", confidence=0.2),
                            ]
                        ),
                    ),
                    FixInfo(
                        col="A",
                        index=2,
                        options=FixValueOptions(
                            options=[FixValue(value="a", confidence=1)]
                        ),
                    ),
                ]
            ),
            2,
            "A",
            "a",
            FixValue(value="a", confidence=1),
        ),
        (
            FixInfoList(
                [
                    FixInfo(
                        col="A",
                        index=1,
                        options=FixValueOptions(
                            options=[
                                FixValue(value="a", confidence=8),
                                FixValue(value="b", confidence=0.2),
                            ]
                        ),
                    ),
                    FixInfo(
                        col="A",
                        index=2,
                        options=FixValueOptions(
                            options=[FixValue(value="c", confidence=1)]
                        ),
                    ),
                ]
            ),
            1,
            "A",
            "c",
            FixValue(value=UNABLE_TO_FIX_PLACEHOLDER, confidence=-inf),
        ),
    ],
)
def test_fix_info_list_find(fix_info_list, index, column, value, expect):
    assert fix_info_list.find(index, column, value) == expect


@pytest.mark.parametrize(
    "fix_info_list, replace_list, expect_fix_info_list",
    [
        (
            FixInfoList(
                [
                    FixInfo(
                        col="A",
                        index=1,
                        options=FixValueOptions(
                            options=[FixValue(value="a", confidence=1)]
                        ),
                    ),
                    FixInfo(
                        col="A",
                        index=2,
                        options=FixValueOptions(
                            options=[FixValue(value="a", confidence=1)]
                        ),
                    ),
                ]
            ),
            [
                FixInfo(
                    col="A",
                    index=1,
                    options=FixValueOptions(
                        options=[FixValue(value="c", confidence=0.2)]
                    ),
                ),
                FixInfo(
                    col="B",
                    index=1,
                    options=FixValueOptions(
                        options=[FixValue(value="c", confidence=0.5)]
                    ),
                ),
            ],
            [
                FixInfo(
                    col="A",
                    index=1,
                    options=FixValueOptions(
                        options=[FixValue(value="c", confidence=0.2)]
                    ),
                ),
                FixInfo(
                    col="A",
                    index=2,
                    options=FixValueOptions(
                        options=[FixValue(value="a", confidence=1)]
                    ),
                ),
                FixInfo(
                    col="B",
                    index=1,
                    options=FixValueOptions(
                        options=[FixValue(value="c", confidence=0.5)]
                    ),
                ),
            ],
        ),
    ],
)
def test_fix_info_list_replace(fix_info_list, replace_list, expect_fix_info_list):
    fix_info_list.replace(*replace_list)

    assert_fix_info_list(fix_info_list, expect_fix_info_list)


@pytest.mark.parametrize(
    "option1, option2, expect_equal",
    [
        (
            FixValueOptions(options=[FixValue(value="c", confidence=0.5)]),
            FixValueOptions(options=[FixValue(value="c", confidence=0.5)]),
            True,
        ),
        (
            FixValueOptions(options=[FixValue(value="c", confidence=0.5)]),
            FixValueOptions(options=[FixValue(value="c", confidence=0.6)]),
            False,
        ),
        (
            FixValueOptions(options=[FixValue(value="c", confidence=0.5)]),
            FixValueOptions(
                options=[
                    FixValue(value="c", confidence=0.5),
                    FixValue(value="d", confidence=0.5),
                ]
            ),
            False,
        ),
        (
            FixValueOptions(options=[FixValue(value="c", confidence=0.5)]),
            FixValueOptions(options=[FixValue(value="b", confidence=0.5)]),
            False,
        ),
    ],
)
def test_fix_info_options_eq(option1, option2, expect_equal):
    assert (option1 == option2) == expect_equal


def test_test_fix_info_list_extend_large_list():
    start = time.time()
    fix_info_list = FixInfoList([])
    for col in range(200):
        size = 3_000
        sub_list = FixInfoList([])
        for i in range(size):
            fix_info = FixInfo(str(col), i, FixValueOptions([]))
            sub_list.append(fix_info)
        fix_info_list.extend(sub_list)
    end = time.time()
    print("Total time cost:", end - start)
