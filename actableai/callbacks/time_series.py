from typing import Dict, Any

import pandas as pd

from actableai.callbacks.base import AAICallback


class AAITimeSeriesCallback(AAICallback):
    """
    TODO write documentation
    """

    def on_tune_model_trained(
        self,
        model_params: Dict[str, Any],
        agg_metrics: Dict[str, float],
        df_item_metrics: pd.DataFrame,
    ):
        """
        TODO write documentation
        """
        pass
