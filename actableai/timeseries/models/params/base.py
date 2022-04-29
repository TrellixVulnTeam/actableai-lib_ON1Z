from hyperopt import hp


class BaseParams:
    """
    Base class for Time Series Model Parameters
    """

    def __init__(
        self,
        model_name,
        is_multivariate_model,
        has_estimator=True,
        handle_feat_static_real=True,
        handle_feat_static_cat=True,
        handle_feat_dynamic_real=False,
        handle_feat_dynamic_cat=False,
    ):
        """
        TODO write documentation
        """
        self.model_name = model_name
        self.is_multivariate_model = is_multivariate_model
        self.has_estimator = has_estimator
        self.handle_feat_static_real = handle_feat_static_real
        self.handle_feat_static_cat = handle_feat_static_cat
        self.handle_feat_dynamic_real = handle_feat_dynamic_real
        self.handle_feat_dynamic_cat = handle_feat_dynamic_cat

    def _hp_param(self, func, param_name, *args, **kwargs):
        """
        TODO write documentation
        """
        return func(f"{self.model_name}_{param_name}", *args, **kwargs)

    def _choice(self, param_name, options):
        """
        TODO write documentation
        """
        if type(options) is not tuple:
            return options
        return self._hp_param(hp.choice, param_name, options)

    def _randint(self, param_name, options):
        """
        TODO write documentation
        """
        if type(options) is not tuple:
            return options
        return self._hp_param(hp.randint, param_name, *options)

    def _uniform(self, param_name, options):
        """
        TODO write documentation
        """
        if type(options) is not tuple:
            return options
        return self._hp_param(hp.uniform, param_name, *options)

    def tune_config(self):
        """
        TODO write documentation
        """
        return {}

    def build_estimator(
        self, *, ctx, device, freq, prediction_length, target_dim, distr_output, params
    ):
        """
        TODO write documentation
        """
        return None

    def build_predictor(self, *, freq, prediction_length, params):
        """
        TODO write documentation
        """
        return None
