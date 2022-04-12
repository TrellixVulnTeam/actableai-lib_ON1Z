import pandas as pd
from functools import reduce


def leaderboard_cross_val(cross_val_leaderboard):
    # Leaderboard
    cross_val_leaderboard = [x.sort_values("model") for x in cross_val_leaderboard]
    only_num_vals = [x.select_dtypes("number") for x in cross_val_leaderboard]
    mean_vals = reduce(lambda x, y: x + y, only_num_vals) / len(only_num_vals)
    leaderboard = pd.DataFrame(
        {
            "model": cross_val_leaderboard[0]["model"],
            "can_infer": cross_val_leaderboard[0]["can_infer"],
        }
    )
    leaderboard = leaderboard.join(mean_vals)
    leaderboard = leaderboard.sort_values("model")
    return leaderboard
