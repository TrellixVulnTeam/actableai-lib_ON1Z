from typing import Optional, Text

from actableai.data_imputation.error_detector.base_error_detector import (
    BaseErrorDetector,
)
from actableai.data_imputation.error_detector.cell_erros import (
    CellErrors,
)
from actableai.data_imputation.error_detector.constraint import Constraints
from actableai.data_imputation.error_detector.smart_column_selector import (
    SmartErrorCellSelector,
)
from actableai.data_imputation.error_detector.validation_constraints_checker import (
    ValidationConstrainsChecker,
)


class ValidationDetector(BaseErrorDetector):
    def __init__(self, constraints: Optional[Constraints] = NotImplemented):
        super(ValidationDetector).__init__()
        self._error_cell_selector = SmartErrorCellSelector()
        self._constraints: Constraints = NotImplemented
        self._validation_checker: ValidationConstrainsChecker = NotImplemented
        self.setup_constraints(constraints)

    def setup_constraints(self, constraints: Constraints):
        self._constraints = constraints
        self._validation_checker = ValidationConstrainsChecker(constraints)

    @classmethod
    def from_constraints(cls, constraints_string: Text):
        return cls(Constraints.parse(constraints_string))

    @property
    def constraints(self):
        return self._constraints

    def detect_cells(self) -> CellErrors:
        self._error_cell_selector.reset()
        candidates = self._validation_checker.detect_most_possible_candidates(self._df)
        for candidate in candidates:
            self._error_cell_selector.append_candidate(candidate)

        errors = CellErrors(self._dftypes)
        for err in self._error_cell_selector.find_actual_errors(
            self._df, self._dftypes, self._validation_checker.mentioned_columns
        ):
            errors.append(err)
        return errors
