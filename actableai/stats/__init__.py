import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.neighbors import KernelDensity
from typing import List, Optional
from tqdm import tqdm


class Stats(object):
    """Class handling calculation of correlation and decorrelation of features."""

    def __init__(self):
        pass

    def _is_numeric(self, df, column):
        return column in df.select_dtypes("number").columns

    def _is_categorical(self, df, column):
        return not self._is_numeric(df, column)

    def corr(
        self,
        df: pd.DataFrame,
        target_col: str,
        target_value: Optional[str] = None,
        p_value: float = 0.05,
        categorical_columns: Optional[List] = None,
        gen_categorical_columns: Optional[List] = None,
    ) -> list:
        """Calculate correlation between target and all other columns.

        Args:
            df: DataFrame containing the target column and other columns.
            target_col: Name of the target column.
            target_value: Value of the target column if categorical.
                Defaults to None.
            p_value: P-value threshold for significance. Defaults to 0.05.

        Raises:
            ValueError: If target_col is not in df.columns or (target_col is
                categorical and target_value is not in df[target_col].unique()).

        Returns:
            list: List containing the correlation between the target and all
        """
        dummy_col_to_original = {}
        for i, cat_col in enumerate(categorical_columns):
            for gen_cat_col in gen_categorical_columns[i]:
                dummy_col_to_original[
                    cat_col + "_" + (gen_cat_col if gen_cat_col is not None else "None")
                ] = cat_col

        if target_value is not None:
            target_col = "_".join([target_col, target_value])
        if target_col not in df.columns:
            raise ValueError(
                "Target column or target value is not in the input dataframe"
            )
        re = []
        spearman_col = df[target_col]
        is_target_col_cat = target_col in dummy_col_to_original.keys()
        for col in list(df.columns):
            if col == target_col or (
                is_target_col_cat
                and col in dummy_col_to_original.keys()
                and dummy_col_to_original[target_col] == dummy_col_to_original[col]
            ):
                df = df.drop(col, axis=1)
        for col in tqdm(list(df.columns)):
            c = spearmanr(spearman_col, df[col], nan_policy="omit")
            if c.pvalue <= p_value:
                original_col = dummy_col_to_original.get(col, col)
                re.append(
                    {
                        "col": [original_col, col[len(original_col) + 1 :]]
                        if col != original_col
                        else col,
                        "corr": c.correlation,
                        "pval": c.pvalue,
                    }
                )
        re.sort(key=lambda r: abs(r["corr"]), reverse=True)
        return re

    def decorrelate(
        self,
        df,
        target_col,
        control_col,
        target_value=None,
        control_value=None,
        kde_steps="auto",
        corr_max=0.05,
        pval_max=0.05,
        kde_steps_=10,
    ) -> list:
        """Re-sample df to de-correlate target_col and control_col.

        Args:
            df: Input DataFrame.
            target_col: Name of the target column.
            control_col: Name of the control column.
            target_value: Value of the target column if categorical. Defaults to None.
            control_value: Value of the control column if categorical. Defaults to None.
            kde_steps: used to compute KDE bandwidth. Higher kde_steps leads to better
                decorrelation but samples with smaller size. Set kde_steps as "auto" to
                search for the smaller value where correlation is insignificant.
            corr_max: Correlation threshold for significance. Defaults to 0.05.
            pval_max: P-value threshold for significance. Defaults to 0.05.
            kde_steps_: Used to compute KDE bandwidth. Higher kde_steps_ leads to better
                decorrelation but samples with smaller size. Defaults to 10.

        Returns:
            list: Sampled indices.
        """
        if kde_steps != "auto":
            kde_steps_ = kde_steps

        if self._is_categorical(df, target_col):
            assert target_value is not None, (
                "target_value must not be None as column '%s' is categorical"
                % target_col
            )

            if self._is_numeric(df, control_col):
                assert control_value is None, (
                    "control_value must be None as column '%s' is continuous"
                    % control_col
                )

                id1 = df[
                    (df[target_col] == target_value) & (df[control_col].notna())
                ].index
                x1 = df[control_col][id1].values.reshape((-1, 1))
                k1 = KernelDensity(bandwidth=(x1.max() - x1.min()) / kde_steps_)
                k1.fit(x1)

                id2 = df[
                    (df[target_col] != target_value) & (df[control_col].notna())
                ].index
                x2 = df[control_col][id2].values.reshape((-1, 1))
                k2 = KernelDensity(bandwidth=(x2.max() - x2.min()) / kde_steps_)
                k2.fit(x2)

                i1, i2 = np.arange(id1.size), np.arange(id2.size)
                np.random.shuffle(i1)
                np.random.shuffle(i2)

                id1, id2 = id1[i1], id2[i2]
                x1, x2 = x1[i1], x2[i2]

                P1 = k1.score_samples(x1.reshape((-1, 1)))
                P2 = k2.score_samples(x1.reshape((-1, 1)))
                id1_ = id1[np.random.rand(id1.size) <= np.exp(np.minimum(P1, P2) - P1)]

                P1 = k1.score_samples(x2.reshape((-1, 1)))
                P2 = k2.score_samples(x2.reshape((-1, 1)))
                id2_ = id2[np.random.rand(id2.size) <= np.exp(np.minimum(P1, P2) - P2)]

                id_ = np.concatenate([id1_, id2_])
                if kde_steps == "auto":
                    corr, pval = spearmanr(
                        (df.loc[id_][target_col] == target_value).astype(int),
                        df.loc[id_][control_col],
                    )
                    if (corr > corr_max) and (pval <= pval_max):
                        return self.decorrelate(
                            df,
                            target_col,
                            control_col,
                            target_value,
                            control_value,
                            kde_steps,
                            corr_max=corr_max,
                            pval_max=pval_max,
                            kde_steps_=kde_steps_ * 2,
                        )

                return id_
            else:
                # Both are categorical
                assert target_value is not None and control_value is not None, (
                    "As both '%s' and '%s' are categorical, neither target_value nor control_value can be None."
                    % (target_col, control_col)
                )
                P = pd.crosstab(
                    df[target_col] == target_value,
                    df[control_col] == control_value,
                    normalize="columns",
                )
                pmin = np.minimum(P[False], P[True])

                iTT = df[
                    (df[target_col] == target_value)
                    & (df[control_col] == control_value)
                ].index
                iTT = iTT[np.random.rand(iTT.size) <= pmin[True] / P[True][True]]

                iTF = df[
                    (df[target_col] == target_value)
                    & (df[control_col] != control_value)
                ].index
                iTF = iTF[np.random.rand(iTF.size) <= pmin[True] / P[False][True]]

                iFT = df[
                    (df[target_col] != target_value)
                    & (df[control_col] == control_value)
                ].index
                iFT = iFT[np.random.rand(iFT.size) <= pmin[False] / P[True][False]]

                iFF = df[
                    (df[target_col] != target_value)
                    & (df[control_col] != control_value)
                ].index
                iFF = iFF[np.random.rand(iFF.size) <= pmin[False] / P[False][False]]
                return np.concatenate([iTT, iTF, iFT, iFF])

        else:
            if self._is_categorical(df, control_col):
                return self.decorrelate(
                    df,
                    control_col,
                    target_col,
                    control_value,
                    target_value,
                    kde_steps=kde_steps,
                )
            else:
                assert control_value is None and target_value is None, (
                    "As both '%s' and '%s' are continous, both target_value and control_value have to be None"
                    % (control_col, target_col)
                )
                df["__corr_tmp__"] = df[control_col] >= df[control_col].median()
                id_ = self.decorrelate(
                    df,
                    target_col,
                    "__corr_tmp__",
                    control_value=True,
                    kde_steps=kde_steps,
                    corr_max=corr_max,
                    pval_max=pval_max,
                    kde_steps_=kde_steps_,
                )
                df.drop(columns=["__corr_tmp__"], inplace=True)
                return id_
