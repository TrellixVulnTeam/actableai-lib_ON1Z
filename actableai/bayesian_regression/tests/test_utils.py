import pandas as pd

from actableai.bayesian_regression.utils import expand_polynomial_categorical


def test_inflate_categorical_polynomial_cate():
    df = pd.DataFrame({"x": ["a", "b", "c"]})
    df_polynomial, orig_dummy_list = expand_polynomial_categorical(df, 1, False)
    assert df_polynomial is not None
    assert orig_dummy_list is not None
    assert df_polynomial.shape == (3, 3)
    assert list(df_polynomial.columns) == ["x_a", "x_b", "x_c"]
    assert orig_dummy_list == ["x_a", "x_b", "x_c"]


def test_inflate_categorical_polynomial_poly():
    df = pd.DataFrame({"x": [1, 2, 3]})
    df_polynomial, orig_dummy_list = expand_polynomial_categorical(df, 4, False)
    assert df_polynomial is not None
    assert orig_dummy_list is not None
    assert df_polynomial.shape == (3, 4)
    assert list(df_polynomial.columns) == ["x", "x^2", "x^3", "x^4"]
    assert orig_dummy_list == ["x"]


def test_inflate_categorical_polynomial_mixed():
    df = pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
    df_polynomial, orig_dummy_list = expand_polynomial_categorical(df, 4, False)
    assert df_polynomial is not None
    assert orig_dummy_list is not None
    assert df_polynomial.shape == (3, 16)
    assert list(df_polynomial.columns) == [
        "x",
        "y_a",
        "y_b",
        "y_c",
        "x^2",
        "x y_a",
        "x y_b",
        "x y_c",
        "x^3",
        "x^2 y_a",
        "x^2 y_b",
        "x^2 y_c",
        "x^4",
        "x^3 y_a",
        "x^3 y_b",
        "x^3 y_c",
    ]
    assert orig_dummy_list == ["x", "y_a", "y_b", "y_c"]


def test_inflate_categorical_polynomial_multi_cat():
    df = pd.DataFrame(
        {
            "x": [1, 2, 3],
            "y": ["a", "b", "c"],
            "z": ["a", "b", "c"],
        }
    )
    df_polynomial, orig_dummy_list = expand_polynomial_categorical(df, 4, False)
    assert df_polynomial is not None
    assert orig_dummy_list is not None
    assert df_polynomial.shape == (3, 55)
    assert list(df_polynomial.columns) == [
        "x",
        "y_a",
        "y_b",
        "y_c",
        "z_a",
        "z_b",
        "z_c",
        "x^2",
        "x y_a",
        "x y_b",
        "x y_c",
        "x z_a",
        "x z_b",
        "x z_c",
        "y_a z_a",
        "y_a z_b",
        "y_a z_c",
        "y_b z_a",
        "y_b z_b",
        "y_b z_c",
        "y_c z_a",
        "y_c z_b",
        "y_c z_c",
        "x^3",
        "x^2 y_a",
        "x^2 y_b",
        "x^2 y_c",
        "x^2 z_a",
        "x^2 z_b",
        "x^2 z_c",
        "x y_a z_a",
        "x y_a z_b",
        "x y_a z_c",
        "x y_b z_a",
        "x y_b z_b",
        "x y_b z_c",
        "x y_c z_a",
        "x y_c z_b",
        "x y_c z_c",
        "x^4",
        "x^3 y_a",
        "x^3 y_b",
        "x^3 y_c",
        "x^3 z_a",
        "x^3 z_b",
        "x^3 z_c",
        "x^2 y_a z_a",
        "x^2 y_a z_b",
        "x^2 y_a z_c",
        "x^2 y_b z_a",
        "x^2 y_b z_b",
        "x^2 y_b z_c",
        "x^2 y_c z_a",
        "x^2 y_c z_b",
        "x^2 y_c z_c",
    ]
    assert orig_dummy_list == ["x", "y_a", "y_b", "y_c", "z_a", "z_b", "z_c"]


def test_inflate_categorical_polynomial_reasonable():
    df = pd.DataFrame(
        {
            "x": [1, 2, 3],
            "y": ["a", "b", "c"],
            "z": ["a", "b", "c"],
            "t": ["d", "e", "f"],
        }
    )
    df_polynomial, orig_dummy_list = expand_polynomial_categorical(df, 2, False)
    assert df_polynomial is not None
    assert orig_dummy_list is not None
    assert df_polynomial.shape == (3, 47)
    assert list(df_polynomial.columns) == [
        "x",
        "y_a",
        "y_b",
        "y_c",
        "z_a",
        "z_b",
        "z_c",
        "t_d",
        "t_e",
        "t_f",
        "x^2",
        "x y_a",
        "x y_b",
        "x y_c",
        "x z_a",
        "x z_b",
        "x z_c",
        "x t_d",
        "x t_e",
        "x t_f",
        "y_a z_a",
        "y_a z_b",
        "y_a z_c",
        "y_a t_d",
        "y_a t_e",
        "y_a t_f",
        "y_b z_a",
        "y_b z_b",
        "y_b z_c",
        "y_b t_d",
        "y_b t_e",
        "y_b t_f",
        "y_c z_a",
        "y_c z_b",
        "y_c z_c",
        "y_c t_d",
        "y_c t_e",
        "y_c t_f",
        "z_a t_d",
        "z_a t_e",
        "z_a t_f",
        "z_b t_d",
        "z_b t_e",
        "z_b t_f",
        "z_c t_d",
        "z_c t_e",
        "z_c t_f",
    ]

    assert orig_dummy_list == [
        "x",
        "y_a",
        "y_b",
        "y_c",
        "z_a",
        "z_b",
        "z_c",
        "t_d",
        "t_e",
        "t_f",
    ]


def test_inflate_categorical_polynomial_normalize():
    df = pd.DataFrame(
        {
            "x": [1, 2, 3],
            "y": ["a", "b", "c"],
            "z": ["a", "b", "c"],
            "t": ["d", "e", "f"],
        }
    )
    df_polynomial, orig_dummy_list = expand_polynomial_categorical(df, 2, True)
    assert df_polynomial is not None
    assert orig_dummy_list is not None
    assert df_polynomial.shape == (3, 47)
    assert list(df_polynomial.columns) == [
        "x",
        "y_a",
        "y_b",
        "y_c",
        "z_a",
        "z_b",
        "z_c",
        "t_d",
        "t_e",
        "t_f",
        "x^2",
        "x y_a",
        "x y_b",
        "x y_c",
        "x z_a",
        "x z_b",
        "x z_c",
        "x t_d",
        "x t_e",
        "x t_f",
        "y_a z_a",
        "y_a z_b",
        "y_a z_c",
        "y_a t_d",
        "y_a t_e",
        "y_a t_f",
        "y_b z_a",
        "y_b z_b",
        "y_b z_c",
        "y_b t_d",
        "y_b t_e",
        "y_b t_f",
        "y_c z_a",
        "y_c z_b",
        "y_c z_c",
        "y_c t_d",
        "y_c t_e",
        "y_c t_f",
        "z_a t_d",
        "z_a t_e",
        "z_a t_f",
        "z_b t_d",
        "z_b t_e",
        "z_b t_f",
        "z_c t_d",
        "z_c t_e",
        "z_c t_f",
    ]
    assert orig_dummy_list == [
        "x",
        "y_a",
        "y_b",
        "y_c",
        "z_a",
        "z_b",
        "z_c",
        "t_d",
        "t_e",
        "t_f",
    ]
    assert ((0 <= df_polynomial) & (df_polynomial <= 1)).all().all()
