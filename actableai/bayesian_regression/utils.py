from typing import Tuple, List
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures

def expand_polynomial_categorical(feature_data:pd.DataFrame, polynomial_degree:int, normalize:bool) -> Tuple[pd.DataFrame, List[str]]:
    """Generates a new DataFrame with extra features :
         - Exponent of numerical features
         - Cross Intersection of variables
         - OneHot encoded values for categorical features

    Args:
        feature_data (pd.DataFrame): DataFrame with the original features. Handles only numerical and categorical features
        polynomial_degree (int): Maximum polynomial degree to generate cross intersection and exponent
        normalize (bool): If we want the Data to be normalized

    Returns:
        Tuple[pd.DataFrame, List[str]]: New DataFrame with generated features, Names of OneHotEncoded variables
    """
    if normalize:
        df_num = feature_data.select_dtypes(include='number')
        df_num = df_num - df_num.min()
        df_num = df_num / df_num.max()
        feature_data[df_num.columns] = df_num

    df_dummies = pd.get_dummies(feature_data)
    df_dummies = pd.DataFrame(
        SimpleImputer(strategy="mean").fit_transform(df_dummies),
        columns=df_dummies.columns
    )

    poly_fit = PolynomialFeatures(polynomial_degree)
    df_polynomial = pd.DataFrame(
        poly_fit.fit_transform(df_dummies),
        columns=poly_fit.get_feature_names(df_dummies.columns)
    ).drop('1', axis=1)

    only_cat_col:pd.DataFrame = feature_data.select_dtypes(exclude='number')
    if not only_cat_col.empty:
        # Dropping the categorical values raised to a certain power.
        regex_cat = [r'{}_.+\^\d+'.format(x) for x in only_cat_col]
        bool_cat = pd.Series(df_polynomial.columns).str.contains('|'.join(regex_cat), regex=True)
        df_polynomial = df_polynomial.drop(df_polynomial.columns[bool_cat], axis=1)

        # Dropping the categorical values with repeated same dummy (y_.+ * y_.+) = (0) because y_a={0, 1}, y_b={0, 1} and y_a != y_b
        regex_rep = [r'({})_.+ \{}_.+'.format(x, i + 1) for i, x in enumerate(only_cat_col)]
        bool_cat = pd.Series(df_polynomial.columns).str.contains('|'.join(regex_rep), regex=True)
        df_polynomial = df_polynomial.drop(df_polynomial.columns[bool_cat], axis=1)

    return df_polynomial, list(df_dummies.columns)
