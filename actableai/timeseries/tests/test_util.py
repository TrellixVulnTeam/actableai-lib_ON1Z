import pandas as pd
import numpy as np
from pytest import mark

from actableai.timeseries.util import findFred


@mark.parametrize("freq", ["H", "S", "T", "N"])
@mark.parametrize("freq_n", ["", "15", "2", "30"])
def test_findFred(freq, freq_n):
    pd_date = pd.Series(
        pd.date_range("25/01/2000 12:32:24", periods=10, freq=freq_n + freq)
    )
    assert findFred(pd_date) == freq_n + freq


@mark.parametrize("freq", ["H", "S", "T", "N"])
@mark.parametrize("freq_n", ["", "15", "2", "30"])
def test_findFred_missing_values(freq, freq_n):
    pd_date = pd.Series(
        pd.date_range("25/01/2000 12:32:24", periods=10, freq=freq_n + freq)
    )
    # Remove two values in the ten values
    pd_date[3] = np.nan
    pd_date[7] = np.nan
    assert findFred(pd_date) == freq_n + freq


def test_findFred_not_enough_values():
    assert (
        findFred(pd.Series(pd.date_range("25/01/2000 12:32:24", periods=2, freq="T")))
        is None
    )


def test_findFred_non_sense():
    assert findFred(pd.Series(["01/02/2012", "03/03/2037", "01/01/1997"])) is None
