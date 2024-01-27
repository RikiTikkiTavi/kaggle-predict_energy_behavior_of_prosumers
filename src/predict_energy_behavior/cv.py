from sklearn.ensemble import VotingRegressor
import predict_energy_behavior.config as config
import predict_energy_behavior.models.joined_model as joined_model
import predict_energy_behavior.utils.common as common
import pandas as pd
from pathlib import Path

from typing import Generator

class TemporalKFold:
    dates: list[pd.Timestamp]

    def __init__(self, dates: list[str]) -> None:
        self.dates = [pd.Timestamp(d) for d in dates]

    def split(
        self, X: pd.DataFrame
    ) -> Generator[tuple[str, pd.DataFrame, pd.DataFrame], None, None]:
        for date in self.dates:
            df_train = X.loc[X["datetime"] <= date]
            df_val = X.loc[X["datetime"] > date]
            fold_name = f"f-{date.year}-{date.month}-{date.day}"
            yield fold_name, df_train, df_val


class MonthlyKFold:
    def __init__(self, months: list[str], max_offset_h: int = 168):
        intervals = []

        for m in months:
            year, month = m.split("-")
            year, month = int(year), int(month)
            t_start = pd.Timestamp(year=year, month=month, day=1)
            t_end = (
                t_start + pd.DateOffset(months=1) - pd.DateOffset(hours=max_offset_h)
            )
            intervals.append((t_start, t_end))

        self.intervals = intervals

    def split(
        self, X: pd.DataFrame
    ) -> Generator[tuple[str, pd.DataFrame, pd.DataFrame], None, None]:
        for val_interval in self.intervals:
            df_val = X.loc[
                (X["datetime"] >= val_interval[0]) & (X["datetime"] <= val_interval[1])
            ]
            df_train = X.loc[~X.index.isin(df_val.index)]
            fold_name = f"f-{val_interval[0].year}-{val_interval[0].month}"
            yield fold_name, df_train, df_val