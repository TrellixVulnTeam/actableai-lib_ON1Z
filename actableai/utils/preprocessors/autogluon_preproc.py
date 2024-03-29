import pandas as pd
from autogluon.features import DatetimeFeatureGenerator, TextNgramFeatureGenerator
from sklearn.feature_extraction.text import CountVectorizer
from pandas import DataFrame
from sklearn.compose import ColumnTransformer
from sklearn.base import TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from nltk.corpus import stopwords
from sklearn.exceptions import NotFittedError

from actableai.utils import get_type_special_no_ag
from actableai.causal.predictors import DataFrameTransformer
from actableai.utils.preprocessors.preprocessing import SKLearnAGFeatureWrapperBase


class CustomeDateTimeFeatureGenerator(DatetimeFeatureGenerator):
    def _generate_features_datetime(self, X: DataFrame) -> DataFrame:
        X_datetime = DataFrame(index=X.index)
        for datetime_feature in self.features_in:
            # TODO: Be aware: When converted to float32 by downstream models, the
            # seconds value will be up to 3 seconds off the true time due to rounding
            # error. If seconds matter, find a separate way to generate (Possibly
            # subtract smallest datetime from all values).
            X_datetime[datetime_feature] = pd.to_datetime(X[datetime_feature])
            X_datetime[datetime_feature + "_year"] = X_datetime[
                datetime_feature
            ].dt.year
            X_datetime[datetime_feature + "_month"] = X_datetime[
                datetime_feature
            ].dt.month_name()
            X_datetime[datetime_feature + "_day"] = X_datetime[datetime_feature].dt.day
            X_datetime[datetime_feature + "_day_of_week"] = X_datetime[
                datetime_feature
            ].dt.day_name()
            X_datetime[datetime_feature] = pd.to_numeric(
                X_datetime[datetime_feature]
            )  # TODO: Use actual date info
            # X_datetime[datetime_feature] =
            #   pd.to_timedelta(X_datetime[datetime_feature]).dt.total_seconds()
            # TODO: Add fastai date features
        return X_datetime


class DMLFeaturizer(TransformerMixin):
    def __init__(self, ct=None) -> None:
        super().__init__()
        self.ct = ct

    def fit(self, X, y=None):
        X = DataFrameTransformer().fit_transform(X)
        type_special = X.apply(get_type_special_no_ag)
        self.ct = ColumnTransformer(
            [
                (
                    OneHotEncoder.__name__,
                    OneHotEncoder(handle_unknown="ignore"),
                    (type_special == "category") | (type_special == "boolean"),
                ),
                (
                    DatetimeFeatureGenerator.__name__,
                    SKLearnAGFeatureWrapperBase(DatetimeFeatureGenerator()),
                    type_special == "datetime",
                ),
                (
                    TextNgramFeatureGenerator.__name__,
                    SKLearnAGFeatureWrapperBase(
                        TextNgramFeatureGenerator(
                            vectorizer=CountVectorizer(stop_words=stopwords.words()),
                            vectorizer_strategy="separate",
                        )
                    ),
                    type_special == "text",
                ),
            ],
            remainder="passthrough",
            sparse_threshold=0,
            verbose_feature_names_out=False,
            verbose=True,
        )
        self.ct.fit(X)
        return self

    def transform(self, X):
        X = DataFrameTransformer().fit_transform(X)
        if self.ct is None:
            raise NotFittedError()
        return self.ct.transform(X)
