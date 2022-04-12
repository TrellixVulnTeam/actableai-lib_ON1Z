import pandas as pd

from actableai.classification.utils import leaderboard_cross_val


def test_leaderboard_cross_val():
    leaderboards = [
        pd.DataFrame(
            {
                "model": ["model1", "model2"],
                "score_val": [0, 1],
                "pred_time_val": [0, 1],
                "fit_time": [0, 1],
                "pred_time_val_marginal": [0, 1],
                "can_infer": [True, True],
                "fit_order": [0, 1],
                "fit_time_marginal": [0, 1],
            }
        ),
        pd.DataFrame(
            {
                "model": ["model1", "model2"],
                "score_val": [1, 2],
                "pred_time_val": [1, 2],
                "fit_time": [1, 2],
                "pred_time_val_marginal": [1, 2],
                "can_infer": [True, True],
                "fit_order": [1, 2],
                "fit_time_marginal": [1, 2],
            }
        ),
    ]
    leaderboard = leaderboard_cross_val(leaderboards)
    assert list(leaderboard["model"]) == ["model1", "model2"]
    assert list(leaderboard["score_val"]) == [0.5, 1.5]
    assert list(leaderboard["pred_time_val"]) == [0.5, 1.5]
    assert list(leaderboard["fit_time"]) == [0.5, 1.5]
    assert list(leaderboard["pred_time_val_marginal"]) == [0.5, 1.5]
    assert list(leaderboard["can_infer"]) == [True, True]
    assert list(leaderboard["fit_order"]) == [0.5, 1.5]
    assert list(leaderboard["fit_time_marginal"]) == [0.5, 1.5]
