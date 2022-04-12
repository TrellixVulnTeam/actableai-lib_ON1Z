import pandas as pd
import numpy as np

from actableai.causal import prepare_sanitize_data


def test_prepare_sanitize_data():
    """test_prepare_sanitize_data"""
    pd_table = pd.DataFrame(
        {
            "x": [x for x in range(4)],
            "y": [x for x in range(4)],
            "z": [x for x in range(4)],
            "t": [str(x) for x in range(4)],
        }
    )
    pd_table = prepare_sanitize_data(pd_table, ["x"], ["y"], ["z"], ["t"])

    assert pd_table is not None
    assert pd_table.shape == (4, 4)


def test_prepare_sanitize_data_drop_treatment():
    """test_prepare_sanitize_data"""
    pd_table = pd.DataFrame(
        {
            "x": [x for x in range(4)],
            "y": [x for x in range(4)],
            "z": [x for x in range(4)],
            "t": [str(x) for x in range(4)],
        }
    )
    pd_table["x"][0] = np.nan
    pd_table = prepare_sanitize_data(pd_table, ["x"], ["y"], ["z"], ["t"])

    assert pd_table is not None
    assert pd_table.shape == (3, 4)


def test_prepare_sanitize_data_drop_effect_modifier():
    """test_prepare_sanitize_data"""
    pd_table = pd.DataFrame(
        {
            "x": [x for x in range(4)],
            "y": [x for x in range(4)],
            "z": [x for x in range(4)],
            "t": [str(x) for x in range(4)],
        }
    )
    pd_table["z"][0] = np.nan
    pd_table = prepare_sanitize_data(pd_table, ["x"], ["y"], ["z"], ["t"])

    assert pd_table is not None
    assert pd_table.shape == (4, 4)


def test_prepare_sanitize_data_drop_comm_causes():
    """test_prepare_sanitize_data"""
    pd_table = pd.DataFrame(
        {
            "x": [x for x in range(4)],
            "y": [x for x in range(4)],
            "z": [x for x in range(4)],
            "t": [str(x) for x in range(4)],
        }
    )
    pd_table["t"][0] = np.nan
    pd_table = prepare_sanitize_data(pd_table, ["x"], ["y"], ["z"], ["t"])

    assert pd_table is not None
    assert pd_table.shape == (4, 4)
