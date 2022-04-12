import os
from typing import Text

import psutil

from actableai.data_imputation.config import logger


def get_memory_usage(stage: Text):
    process = psutil.Process(os.getpid())
    logger.debug(
        f"Memory usage in stage - '{stage}':",
        process.memory_info().rss * 9.31 * 10**-10,
        "GB",
    )
