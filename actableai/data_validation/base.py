import pandas as pd
import abc
from typing import List

MINIMUM_NUMBER_OF_SAMPLE = 20
CLASSIFICATION_MINIMUM_NUMBER_OF_CLASS = 2
UNIQUE_CATEGORY_THRESHOLD = 10
REGRESSION_ANALYTIC = "regression"
CLASSIFICATION_ANALYTIC = "classification"
CLASSIFICATION_MINIMUM_NUMBER_OF_CLASS_SAMPLE = 2
EXPLAIN_SAMPLES_UNIQUE_CATEGORICAL_LIMIT = 100
CAUSAL_INFERENCE_CATEGORICAL_MINIMUM_TREATMENT = 20
POLYNOMIAL_INFLATE_COLUMN_LIMIT = 1000


class CheckLevels:
    CRITICAL = "CRITICAL"
    WARNING = "WARNING"


class ValidationStatus:
    HEALTHY = "HEALTHY"
    UNHEALTHY = "UNHEALTHY"
    CRITICAL = "CRITICAL"


class CheckResult:
    def __init__(self, name: str, message: str, level: str):
        self.message = message
        self.name = name
        self.level = level

    def __str__(self):
        return repr({"message": self.message, "name": self.name, "level": self.level})

    def __repr__(self):
        return repr({"message": self.message, "name": self.name, "level": self.level})


class IChecker(metaclass=abc.ABCMeta):
    def __init__(self, name: str):
        self.name = name

    @abc.abstractmethod
    def check(self, df: pd.DataFrame) -> CheckResult:
        pass
