from sklearn.metrics import recall_score, accuracy_score
from statistics import mean
import pandas as pd


def evaluate(data_name, new_version: int = 0):
    CLEAN_CSV = f"../experiments/data/{data_name}/{data_name}.csv"
    MY_FIXED_CSV = f"../experiments/data/{data_name}/repaired-from-my{f'-version-{new_version}'  if new_version > 0 else ''}.csv"
    HOLOCLEAN_FIXED_CSV = f"../experiments/data/{data_name}/repaired-from-holoclean.csv"

    clean_csv = pd.read_csv(CLEAN_CSV)
    hc_fixed_csv = pd.read_csv(HOLOCLEAN_FIXED_CSV)
    my_fixed_csv = pd.read_csv(MY_FIXED_CSV)

    y_true = []
    y_pred_hc = []
    y_pred_my = []
    overall_recall_score_hc = []
    overall_recall_score_my = []
    for col in clean_csv.columns:
        # there are some additional 1 empty line from the holoclean fix, filter them out
        true_column = clean_csv[col]
        hc_column = hc_fixed_csv[col]
        my_column = my_fixed_csv[col]
        hc_column = hc_column[: len(true_column)]
        my_column = my_column[: len(true_column)]

        if sum(pd.isnull(true_column)) == len(true_column):
            continue

        # filter out NaN values from gold data
        non_null_indexes = ~pd.isnull(true_column)
        true_column = true_column[non_null_indexes]
        hc_column = hc_column[non_null_indexes]
        my_column = my_column[non_null_indexes]

        y_true.extend(true_column)
        y_pred_hc.extend(hc_column)
        y_pred_my.extend(my_column)

        overall_recall_score_hc.append(
            recall_score(
                y_true=true_column,
                y_pred=hc_column,
                average="macro",
            )
        )
        overall_recall_score_my.append(
            recall_score(y_true=true_column, y_pred=my_column, average="macro")
        )

    overall_precision_score_hc = accuracy_score(y_true=y_true, y_pred=y_pred_hc)
    overall_precision_score_my = accuracy_score(y_true=y_true, y_pred=y_pred_my)

    print(f"Overall Precision:")
    print(f"  Holoclean: {overall_precision_score_hc}")
    print(f"  My: {overall_precision_score_my}")

    print(f"Average Recall:")
    print(f"  Holoclean: {mean(overall_recall_score_hc)}")
    print(f"  My: {mean(overall_recall_score_my)}")
