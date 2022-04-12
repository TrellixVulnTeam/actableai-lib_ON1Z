from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from math import inf
from typing import Union, List, Optional, Iterable, Dict

from actableai.data_imputation.config import UNABLE_TO_FIX_PLACEHOLDER
from actableai.data_imputation.meta.column import ColumnName

ValueType = Union[str, int, bool, float, complex]


@dataclass(frozen=True)
class FixValue:
    value: ValueType
    confidence: float


@dataclass(frozen=True)
class FixValueOptions:
    options: List[FixValue]

    def __len__(self):
        return len(self.options)


@dataclass(unsafe_hash=False, eq=False, frozen=True)
class FixInfo:
    col: str
    index: int
    options: FixValueOptions

    def __eq__(self, other):
        if not isinstance(other, FixInfo):
            return False

        return self.col == other.col and self.index == other.index

    def __hash__(self):
        return hash(f"{self.col}_{self.index}")

    @property
    def sorted_options(self) -> List[FixValue]:
        return list(
            sorted(self.options.options, key=lambda x: x.confidence, reverse=True)
        )

    @property
    def best_guess(self) -> Union[str, int, bool, float, complex]:
        return self.sorted_options[0].value


class FixInfoList:
    def __init__(self, fix_info_list: List[FixInfo] = ()):
        self.__info_list_dict: Dict[ColumnName, List[FixInfo]] = defaultdict(list)
        self.__info_index_to_list_index_dict: Dict[
            ColumnName, Dict[int, int]
        ] = defaultdict(dict)
        for fix_info in fix_info_list:
            self.append(fix_info)

    def __iter__(self) -> FixInfo:
        for column in self.columns:
            bucket = self.__info_list_dict[column]
            for fix_info in bucket:
                yield fix_info

    def __repr__(self):
        return ", ".join(map(str, self))

    def __str__(self):
        return self.__repr__()

    def __len__(self):
        return sum(map(len, self.__info_list_dict.values()))

    def __eq__(self, other):
        if not isinstance(other, FixInfoList):
            return False

        this = list(sorted(self, key=lambda x: f"{x.col}_{x.index}"))
        other = list(sorted(other, key=lambda x: f"{x.col}_{x.index}"))
        if len(this) != len(other):
            return False

        return all([x == y for x, y in zip(this, other)])

    def __hash__(self):
        return hash(id(self))

    @property
    def index(self) -> Iterable[int]:
        return set([v.index for v in self])

    @property
    def columns(self) -> Iterable[ColumnName]:
        return set(self.__info_list_dict.keys())

    def append(self, fix_info: FixInfo):
        bucket = self.__info_list_dict.get(fix_info.col, [])
        index_map_in_bucket = self.__info_index_to_list_index_dict.get(
            fix_info.col, dict()
        )
        if len(bucket) == 0:
            self.__info_list_dict[fix_info.col].append(fix_info)
            self.__info_index_to_list_index_dict[fix_info.col][fix_info.index] = 0
            return

        if fix_info.index in index_map_in_bucket.keys():
            index = index_map_in_bucket[fix_info.index]
            bucket[index] = fix_info
        else:
            index_map_in_bucket[fix_info.index] = len(bucket)
            bucket.append(fix_info)

    def extend(self, fix_info_list: "FixInfoList"):
        for fix_info in fix_info_list:
            self.append(fix_info)

    def get_item(self, index: int, column: ColumnName) -> Optional[FixInfo]:
        for info in self:
            if info.index == index and info.col == column:
                return info
        else:
            return None

    def get_via_index(self, index: int) -> Iterable[FixInfo]:
        return filter(lambda x: x.index == index, self)

    def replace(self, *fix_infos: FixInfo):
        for v in fix_infos:
            self.append(v)

    @lru_cache(maxsize=100)
    def find(self, index: int, column: ColumnName, value: ValueType) -> FixValue:
        unable_to_fix = FixValue(UNABLE_TO_FIX_PLACEHOLDER, -inf)

        fix_info = self.get_item(index, column)
        if fix_info is None:
            return unable_to_fix

        for option in fix_info.sorted_options:
            if option.value == value:
                return option
        else:
            return unable_to_fix
