import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KernelDensity

class Stats(object):

    def __init__(self):
        pass

    def _is_numeric(self, df, column):
        return column in df.select_dtypes("number").columns

    def _is_categorical(self, df, column):
        return not self._is_numeric(df, column)

    def corr(self, df, target_col, target_value=None, p_value=0.05):
        """ Return correlations of target_col and other columns in a dataframe. Categorical values are converted using
        one-hot-encoding.

        @return a Pandas dataframe whose index is column names and columns are correlation and p-values respectively.

        """
        categorical_columns = df.columns[df.dtypes == object]
        train_df = df[categorical_columns].fillna('NaN')
        enc = OneHotEncoder()
        enc.fit(train_df)
        dummies = pd.DataFrame.sparse.from_spmatrix(
            enc.transform(train_df),
            columns=enc.get_feature_names(categorical_columns))
        dummy_col_to_original = {}
        for i in range(len(categorical_columns)):
            for t in enc.categories_[i]:
                dummy_col_to_original[categorical_columns[i] + '_' + str(t)] = categorical_columns[i]

        x = pd.concat([df.drop(columns=categorical_columns), dummies], axis=1)
        if target_value is not None:
            target_col = "_".join([target_col, target_value])
        if target_col not in x.columns:
            raise ValueError("Target column or target value is not in the input dataframe")
        re = []
        spearman_col = x[target_col]
        is_target_col_cat = target_col in dummy_col_to_original.keys()
        for col in x.columns:
            if col == target_col or (is_target_col_cat and col in dummy_col_to_original.keys()
                                    and dummy_col_to_original[target_col] == dummy_col_to_original[col]):
                x = x.drop(col, axis=1)
        for col in x.columns:
            c = spearmanr(spearman_col, x[col], nan_policy="omit")
            if c.pvalue <= p_value:
                original_col = dummy_col_to_original.get(col, col)
                re.append({
                    "col": [original_col, col[len(original_col) + 1:]] if col != original_col else col,
                    "corr": c.correlation,
                    "pval": c.pvalue
                })
        re.sort(key=lambda r: abs(r["corr"]), reverse=True)
        return re

    def decorrelate(self, df, target_col, control_col, target_value=None, control_value=None, kde_steps="auto",
                    corr_max=0.05, pval_max=0.05, kde_steps_=10):
        """ Re-sample df to de-correlate target_col and control_col.

        @kde_step: this is used to compute KDE bandwidth. Higher kde_steps leads to better decorrelation but samples
        with smaller size. Set kde_steps as "auto" to search for the smaller value where correlation is insignificant.

        @return the sampled indices.

        """
        if kde_steps != "auto":
            kde_steps_ = kde_steps

        if self._is_categorical(df, target_col):
            assert target_value is not None, \
                "target_value must not be None as column '%s' is categorical" % target_col

            if self._is_numeric(df, control_col):
                assert control_value is None, \
                    "control_value must be None as column '%s' is continuous" % control_col

                id1 = df[(df[target_col]==target_value) & (df[control_col].notna())].index
                x1 = df[control_col][id1].values.reshape((-1, 1))
                k1 = KernelDensity(bandwidth=(x1.max() - x1.min())/kde_steps_)
                k1.fit(x1)

                id2 = df[(df[target_col]!=target_value) & (df[control_col].notna())].index
                x2 = df[control_col][id2].values.reshape((-1, 1))
                k2 = KernelDensity(bandwidth=(x2.max() - x2.min())/kde_steps_)
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
                    corr, pval = spearmanr((df.loc[id_][target_col]==target_value).astype(int),
                                           df.loc[id_][control_col])
                    if (corr > corr_max) and (pval <= pval_max):
                        return self.decorrelate(df, target_col, control_col, target_value, control_value, kde_steps,
                                                corr_max=corr_max, pval_max=pval_max, kde_steps_=kde_steps_*2)

                return id_
            else:
                # Both are categorical
                assert target_value is not None and control_value is not None,\
                    "As both '%s' and '%s' are categorical, neither target_value nor control_value can be None." \
                    % (target_col, control_col)
                P = pd.crosstab(df[target_col]==target_value, df[control_col]==control_value, normalize="columns")
                pmin = np.minimum(P[False], P[True])

                iTT = df[(df[target_col]==target_value) & (df[control_col]==control_value)].index
                iTT = iTT[np.random.rand(iTT.size) <= pmin[True]/P[True][True]]

                iTF = df[(df[target_col]==target_value) & (df[control_col]!=control_value)].index
                iTF = iTF[np.random.rand(iTF.size) <= pmin[True]/P[False][True]]

                iFT = df[(df[target_col]!=target_value) & (df[control_col]==control_value)].index
                iFT = iFT[np.random.rand(iFT.size) <= pmin[False]/P[True][False]]

                iFF= df[(df[target_col]!=target_value) & (df[control_col]!=control_value)].index
                iFF = iFF[np.random.rand(iFF.size) <= pmin[False]/P[False][False]]
                return np.concatenate([iTT, iTF, iFT, iFF])

        else:
            if self._is_categorical(df, control_col):
                return self.decorrelate(df, control_col, target_col, control_value, target_value, kde_steps=kde_steps)
            else:
                assert control_value is None and target_value is None, \
                    "As both '%s' and '%s' are continous, both target_value and control_value have to be None" \
                    % (control_col, target_col)
                df["__corr_tmp__"] = df[control_col] >= df[control_col].median()
                id_ = self.decorrelate(df, target_col, "__corr_tmp__", control_value=True, kde_steps=kde_steps,
                                       corr_max=corr_max, pval_max=pval_max, kde_steps_=kde_steps_)
                df.drop(columns=["__corr_tmp__"], inplace=True)
                return id_

