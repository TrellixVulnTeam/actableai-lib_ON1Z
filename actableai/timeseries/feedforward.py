from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.dataset.field_names import FieldName
from gluonts.time_feature import time_features_from_frequency_str
from gluonts.transform import (
    Chain,
    ExpectedNumInstanceSampler,
    AddObservedValuesIndicator,
    AddTimeFeatures,
    AddAgeFeature,
    InstanceSplitter,
    Transformation,
    VstackFeatures,
)


class FeedForwardEstimator(SimpleFeedForwardEstimator):
    """In addition to the overridden model, this model:
    - Works with inputs with NaN values.
    - Adds time and age features.
    """

    def create_transformation(self) -> Transformation:
        return Chain(
            [
                AddObservedValuesIndicator(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.OBSERVED_VALUES,
                    dtype=self.dtype,
                ),
                AddTimeFeatures(
                    start_field=FieldName.START,
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_TIME,
                    time_features=time_features_from_frequency_str(self.freq),
                    pred_length=self.prediction_length,
                ),
                AddAgeFeature(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_AGE,
                    pred_length=self.prediction_length,
                    log_scale=True,
                    dtype=self.dtype,
                ),
            ]
        )
