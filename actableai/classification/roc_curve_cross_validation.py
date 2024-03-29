import numpy as np
from typing import Dict


def cross_validation_curve(
    cross_val_auc_curves: Dict,
    x: str = "False Positive Rate",
    y: str = "True Positive Rate",
    negative_label: bool = True,
) -> Dict:
    """Computes the combined curves for ROC and Precision-Recall curves when using
        cross-validation.

    Args:
        roc_curves_dictionnary: A dictionnary containing the ROC curves for each classifier
    Returns:
        A dictionnary containing the combined ROC curves for each classifier
    """
    thresholds = np.linspace(0, 1, 100)
    x_list = []
    y_list = []
    for i, _ in enumerate(cross_val_auc_curves["thresholds"]):
        interp_x = np.interp(
            thresholds,
            cross_val_auc_curves["thresholds"][i],
            cross_val_auc_curves[x][i],
        )
        interp_y = np.interp(
            thresholds,
            cross_val_auc_curves["thresholds"][i],
            cross_val_auc_curves[y][i],
        )
        x_list.append(interp_x)
        y_list.append(interp_y)
    x_mean = np.mean(x_list, axis=0)
    y_mean = np.mean(y_list, axis=0)
    y_stdrs = np.std(y_list, axis=0)
    if negative_label:
        return {
            x: x_mean,
            y: y_mean,
            f"{y} stderr": y_stdrs,
            "thresholds": thresholds,
            "positive_label": cross_val_auc_curves["positive_label"][0],
            "negative_label": cross_val_auc_curves["negative_label"][0],
        }
    else:
        return {
            x: x_mean,
            y: y_mean,
            f"{y} stderr": y_stdrs,
            "thresholds": thresholds,
            "positive_label": cross_val_auc_curves["positive_label"][0],
        }
