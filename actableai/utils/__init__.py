import os
import uuid
from copy import deepcopy

import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame


class AutogluonShapWrapper:
    def __init__(self, predictor, feature_names):
        self.ag_model = predictor
        self.feature_names = feature_names

    def predict(self, X):
        if isinstance(X, pd.Series):
            X = X.values.reshape(1, -1)
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names)
        return self.ag_model.predict(X)

    def predict_proba(self, X):
        if isinstance(X, pd.Series):
            X = X.values.reshape(1, -1)
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names)
        return self.ag_model.predict_proba(X)


class AnchorWrapper(object):

    def __init__(self, predictor, columns, cat_columns, ordinal_encoder=None):
        self.predictor = predictor
        self.columns = columns
        self.ordinal_encoder = ordinal_encoder
        self.cat_columns = cat_columns

    def predict_proba(self, X):
        df = pd.DataFrame(X, columns=self.columns)
        if len(self.cat_columns) > 0:
            df[self.cat_columns] = self.ordinal_encoder.inverse_transform(df[self.cat_columns])
        return self.predictor.predict_proba(df)

    def predict(self, X):
        df = pd.DataFrame(X, columns=self.columns)
        if len(self.cat_columns) > 0:
            df[self.cat_columns] = self.ordinal_encoder.inverse_transform(df[self.cat_columns])
        return self.predictor.predict(df)


def fill_na(df, fillna_dict=None, fill_median=True):
    import numpy as np
    if fillna_dict == None:
        fillna_dict = {
            object: '',
            int: np.nan,
            float: np.nan
        }

    cat_cols = df.select_dtypes(exclude=["number"]).columns
    ordinal_cols = df.select_dtypes(include=["number"]).columns
    df[cat_cols] = df[cat_cols].fillna(fillna_dict[object])
    if fill_median:
        for col in ordinal_cols:
            df[col] = df[col].fillna(df[col].median())
    else:
        df[ordinal_cols] = df[ordinal_cols].fillna(fillna_dict[int])

    return df

def explain_predictions(df, cat_cols, explainer, encoder=None, threshold=0.80):
    df_processed = preprocess_data_for_anchor(df, cat_cols, encoder)
    predict_explains = []
    origin_indices = df.index.values
    for idx, row in enumerate(df_processed):
        explanation = explainer.explain(row, threshold=threshold)
        explanation_dict = {}
        explanation_dict["anchor"] = explanation.anchor
        explanation_dict['precision'] = explanation.precision
        explanation_dict['coverage'] = explanation.coverage
        explanation_dict['index'] = int(origin_indices[idx])
        predict_explains.append(explanation_dict)
        print(explanation_dict)
    return predict_explains

def handle_datetime_features(df):
    datetime_cols = df.select_dtypes(include=["datetime"]).columns
    for col in datetime_cols:
        df[col] = pd.to_numeric(df[col])
    return df

def handle_boolean_features(df):
    from pandas.api.types import infer_dtype
    for col in df.columns:
        dtype = infer_dtype(df[col])
        if dtype == "boolean":
            df[col][pd.isnull(df[col])] = ""
            df[col] = df[col].astype(str)
            df[col][df[col]==""] = np.NaN
    return df

def preprocess_data_for_anchor(df, cat_cols, encoder=None):
    if len(cat_cols) > 0:
        df[cat_cols] = encoder.transform(df[cat_cols])
    return df.values

def preprocess_dataset(df):
    df = handle_datetime_features(df)
    df = handle_boolean_features(df)
    df = fill_na(df)
    return df

def create_explainer(df, predictor, cat_map, encoder=None, ncpu=1):
    from alibi.explainers import AnchorTabular, DistributedAnchorTabular
    from actableai.utils import AnchorWrapper

    cat_cols = df.columns[list(cat_map.keys())]
    anchor_wrapper = AnchorWrapper(predictor, df.columns, cat_cols, encoder)
    if ncpu > 1:
        explainer = DistributedAnchorTabular(
            anchor_wrapper.predict_proba,
            df.columns,
            categorical_names=cat_map
        )
        explainer.fit(preprocess_data_for_anchor(df, cat_cols, encoder), disc_perc=[25, 50, 75], ncpu=ncpu)
    else:
        explainer = AnchorTabular(
            anchor_wrapper.predict_proba,
            df.columns,
            categorical_names=cat_map
        )
        explainer.fit(preprocess_data_for_anchor(df, cat_cols, encoder), disc_perc=[25, 50, 75])
    return explainer


def get_type_special(X: pd.Series) -> str:
    from autogluon.core.features.infer_types import check_if_datetime_as_object_feature, check_if_nlp_feature
    from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype, infer_dtype
    # from autogluon.core.utils import infer_problem_type
    import numpy as np

    type_special = "unknown"
    if len(X)>0:
        if 'mixed' in infer_dtype(X):
            type_special = 'mixed'
        elif infer_dtype(X) == 'boolean':
            type_special = 'boolean'
        elif is_datetime64_any_dtype(X):
            type_special = 'datetime'
        elif check_if_datetime_as_object_feature(X):
            type_special = 'datetime'
        elif check_if_nlp_feature(X):
            type_special = 'text'
        elif X.dtype == np.dtype('O'):
            type_special = "category"
        elif check_if_integer_feature(X):
            type_special = 'integer'
        elif is_numeric_dtype(X):
            type_special = "numeric"
    elif len(X)==0:
        type_special="empty"
    return type_special

def get_type_special_no_ag(X: pd.Series) -> str:
    """
    From autogluon library
    TODO improve
    """
    from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype, infer_dtype
    import numpy as np
    import logging

    def get_type_family_raw(dtype) -> str:
        """From dtype, gets the dtype family."""
        try:
            if dtype.name == 'category':
                return 'category'
            if 'datetime' in dtype.name:
                return 'datetime'
            elif np.issubdtype(dtype, np.integer):
                return 'int'
            elif np.issubdtype(dtype, np.floating):
                return 'float'
        except Exception as err:
            logging.error(
                f'Warning: dtype {dtype} is not recognized as a valid dtype by numpy! AutoGluon may incorrectly handle this feature...')
            logging.error(err)

        if dtype.name in ['bool', 'bool_']:
            return 'bool'
        elif dtype.name in ['str', 'string', 'object']:
            return 'object'
        else:
            return dtype.name

    def check_if_datetime_as_object_feature(X: pd.Series) -> bool:
        type_family = get_type_family_raw(X.dtype)
        if X.isnull().all():
            return False
        if type_family != 'object':
            return False
        try:
            X.apply(pd.to_numeric)
        except:
            try:
                X.apply(pd.to_datetime)
                return True
            except:
                return False
        else:
            return False

    def check_if_nlp_feature(X: pd.Series) -> bool:
        type_family = get_type_family_raw(X.dtype)
        if type_family != 'object':
            return False
        X_unique = X.unique()
        num_unique = len(X_unique)
        num_rows = len(X)
        unique_ratio = num_unique / num_rows
        if unique_ratio <= 0.01:
            return False
        try:
            avg_words = pd.Series(X_unique).str.split().str.len().mean()
        except AttributeError:
            return False
        if avg_words < 3:
            return False

        return True

    def check_if_integer_feature(X: pd.Series):
        import numpy as np
        clean_X = X.dropna()
        return np.array_equal(clean_X.values, clean_X.values.astype(int))

    type_special = "unknown"
    if len(X) > 0:
        if 'mixed' in infer_dtype(X):
            type_special = 'mixed'
        elif infer_dtype(X) == 'boolean':
            type_special = 'boolean'
        elif is_datetime64_any_dtype(X):
            type_special = 'datetime'
        elif check_if_datetime_as_object_feature(X):
            type_special = 'datetime'
        elif check_if_nlp_feature(X):
            type_special = 'text'
        elif X.dtype == np.dtype('O'):
            type_special = "category"
        elif check_if_integer_feature(X):
            type_special = 'integer'
        elif is_numeric_dtype(X):
            type_special = "numeric"
    elif len(X) == 0:
        type_special = "empty"
    return type_special

def check_if_integer_feature(X: pd.Series):
    import numpy as np
    clean_X = X.dropna()
    return np.array_equal(clean_X.values, clean_X.values.astype(int))

def gen_anchor_explanation(anchor, total_samples):
    return "{}% of similar generated samples that satisfy {} are predicted to belong to this class".format(
        int(anchor["precision"] * 100), " and ".join(anchor["anchor"]))

def memory_efficient_hyperparameters():
    from autogluon.tabular.configs.hyperparameter_configs import hyperparameter_config_dict
    # Returns autogluon tabular predictor's hyperparameters without the heavy-memory models

    hyperparameters = deepcopy(hyperparameter_config_dict["default"])
    if "NN" in hyperparameters:
        del hyperparameters["NN"]

    # Text models
    hyperparameters["FASTTEXT"] = {}
    hyperparameters["AG_TEXT_NN"] = {}

    return hyperparameters

def preprocess_data_for_shap(X:DataFrame):
    from autogluon.features.generators import DatetimeFeatureGenerator
    shap_data = X.copy()
    datetime_columns = X.select_dtypes(include=['datetime']).columns
    shap_data[datetime_columns] = DatetimeFeatureGenerator().fit_transform(shap_data[datetime_columns])
    return shap_data

def fast_categorical_hyperparameters():
    from autogluon.tabular.configs.hyperparameter_configs import hyperparameter_config_dict
    # Returns autogluon tabular predictor's hyperparameters for fast training with cat vars

    hyperparameters = deepcopy(hyperparameter_config_dict["default"])
    if "NN" in hyperparameters:
        del hyperparameters["NN"]
    if "GBM" in hyperparameters:
        del hyperparameters["GBM"]
    if "XGB" in hyperparameters:
        del hyperparameters["XGB"]

    return hyperparameters

def debiasing_hyperparameters():
    return {
        "LR": {}
    }

def debiasing_feature_generator_args():
    return {
        "enable_numeric_features": True,
        "enable_categorical_features": True,
        "enable_datetime_features": True,
        "enable_text_special_features": False,
        "enable_text_ngram_features": False,
        "enable_raw_text_features": False,
        "enable_vision_features": True
    }

def random_directory(path=""):
    """
    Create random directory,
    """
    uid = uuid.uuid4()
    directory = os.path.join(path, str(uid))

    return directory
