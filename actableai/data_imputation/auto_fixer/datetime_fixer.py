import pandas as pd
from datetime import datetime

from actableai.data_imputation.auto_fixer.auto_fixer import AutoFixer
from actableai.data_imputation.auto_fixer.fix_info import (
    FixInfoList,
    FixInfo,
    FixValueOptions,
    FixValue,
)

from actableai.data_imputation.error_detector import CellErrors
from actableai.data_imputation.meta.column import RichColumnMeta


class DatetimeFixer(AutoFixer):
    def fix(
        self,
        df: pd.DataFrame,
        all_errors: CellErrors,
        current_column: RichColumnMeta,
    ) -> FixInfoList:
        df[current_column.name] = pd.to_datetime(df[current_column.name])
        df_to_fix = df.copy()

        fix_info_list = FixInfoList()

        try:
            # try infer freqency from 3 consecutive non-nan datetime
            dt_series = df_to_fix[current_column.name]
            valid_series_idx = (
                dt_series.iloc[
                    dt_series.index[pd.notna(dt_series).rolling(window=3).sum().ge(3)]
                    - 2
                ]
                .head(1)
                .index.to_list()[0]
            )

            valid_datatime_series = dt_series.iloc[
                valid_series_idx : valid_series_idx + 3
            ]
            freq = pd.infer_freq(valid_datatime_series)
            if freq is not None:
                if not (pd.isna(dt_series.values[0]) or pd.isna(dt_series.values[-1])):
                    df_to_fix.set_index(current_column.name, inplace=True)
                    series_with_fix = df_to_fix.resample(freq).first().index.to_series()
                else:
                    ts_series = dt_series.apply(
                        lambda x: datetime.timestamp(x) if pd.notna(x) else x
                    )
                    nan_indices = ts_series[pd.isna(ts_series)].index.to_list()
                    while len(nan_indices) > 0:
                        for idx in nan_indices:
                            if (
                                (
                                    idx + 1 not in nan_indices
                                    and idx - 1 not in nan_indices
                                )
                                and idx + 1 < len(dt_series)
                                and idx - 1 >= 0
                            ):
                                ts_series[idx] = (
                                    ts_series[idx + 1] + ts_series[idx - 1]
                                ) / 2
                                nan_indices.remove(idx)
                            elif (
                                idx + 1 not in nan_indices
                                and idx + 2 not in nan_indices
                            ) and idx + 2 < len(dt_series):
                                ts_series[idx] = ts_series[idx + 1] - (
                                    ts_series[idx + 2] - ts_series[idx + 1]
                                )
                                nan_indices.remove(idx)
                            elif (
                                idx - 1 not in nan_indices
                                and idx - 2 not in nan_indices
                            ) and idx - 2 >= 0:
                                ts_series[idx] = ts_series[idx - 1] - (
                                    ts_series[idx - 2] - ts_series[idx - 1]
                                )
                                nan_indices.remove(idx)

                        nan_indices = ts_series[pd.isna(ts_series)].index.to_list()

                    series_with_fix = ts_series.apply(datetime.fromtimestamp)

                for err in all_errors[current_column.name]:
                    fix_info_list.append(
                        FixInfo(
                            col=current_column.name,
                            index=err.index,
                            options=FixValueOptions(
                                options=[FixValue(series_with_fix.iloc[err.index], 1)]
                            ),
                        )
                    )
                return fix_info_list
            else:
                return fix_info_list
        except Exception as e:
            return fix_info_list
