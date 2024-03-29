import abc
import pandas as pd
from typing import Optional

MINIMUM_NUMBER_OF_SAMPLE = 20
CLASSIFICATION_MINIMUM_NUMBER_OF_CLASS = 2
UNIQUE_CATEGORY_THRESHOLD = 20
REGRESSION_ANALYTIC = "regression"
CLASSIFICATION_ANALYTIC = "classification"
CLASSIFICATION_MINIMUM_NUMBER_OF_CLASS_SAMPLE = 2
EXPLAIN_SAMPLES_UNIQUE_CATEGORICAL_LIMIT = 100
CAUSAL_INFERENCE_CATEGORICAL_MINIMUM_TREATMENT = 20
POLYNOMIAL_INFLATE_COLUMN_LIMIT = 1000
CORRELATION_MINIMUM_NUMBER_OF_SAMPLE = 3
CAUSAL_FEATURE_SELECTION_CATEGORICAL_MAX_UNIQUES = 10


class CheckLevels:
    """Class holding the check levels for data validation"""

    CRITICAL = "CRITICAL"
    WARNING = "WARNING"


class ValidationStatus:
    HEALTHY = "HEALTHY"
    UNHEALTHY = "UNHEALTHY"
    CRITICAL = "CRITICAL"


class CheckResult:
    def __init__(self, name: str, message: str, level: str):
        """Result of a check.

        Args:
            name: Name of the check.
            message: Message of the check.
            level: Level of the check.
        """
        self.message = message
        self.name = name
        self.level = level

    def __str__(self):
        return repr({"message": self.message, "name": self.name, "level": self.level})

    def __repr__(self):
        return repr({"message": self.message, "name": self.name, "level": self.level})


class IChecker(metaclass=abc.ABCMeta):
    """Abstract class for Checker"""

    def __init__(self, name: str):
        self.name = name

    @abc.abstractmethod
    def check(self, df: pd.DataFrame) -> Optional[CheckResult]:
        pass
