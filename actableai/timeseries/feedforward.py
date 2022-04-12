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
                    dummy_value=self.distr_output.value_in_support,
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
                VstackFeatures(
                    output_field=FieldName.FEAT_TIME,
                    input_fields=[FieldName.FEAT_TIME, FieldName.FEAT_AGE],
                ),
                InstanceSplitter(
                    target_field=FieldName.TARGET,
                    is_pad_field=FieldName.IS_PAD,
                    start_field=FieldName.START,
                    forecast_start_field=FieldName.FORECAST_START,
                    train_sampler=ExpectedNumInstanceSampler(num_instances=1),
                    past_length=self.context_length,
                    future_length=self.prediction_length,
                    time_series_fields=[
                        FieldName.FEAT_TIME,
                        FieldName.OBSERVED_VALUES,
                    ],
                ),
            ]
        )
