import datetime
from itertools import chain
from pathlib import Path
import geopy
import pandas as pd
import polars as pl
import numpy as np
import holidays
from predict_energy_behavior.data.data_storage import DataStorage
import predict_energy_behavior.config as config
from shapely.geometry import Point, Polygon
import geopy.distance
from global_land_mask import globe
from datetime import timedelta


def calculate_relative_humidity(T: pl.Expr, D: pl.Expr) -> pl.Expr:
    """
    Calculate the relative humidity based on temperature and dewpoint.
    T: Temperature in Celsius
    D: Dewpoint in Celsius
    """
    return 100 * (
        ((17.625 * D) / (243.04 + D)).exp() / ((17.625 * T) / (243.04 + T)).exp()
    )


def wind_factor(WS: pl.Expr, threshold=2.0) -> pl.Expr:
    """
    Calculate the wind factor for fog formation.
    WS: Wind Speed in m/s
    threshold: Wind speed threshold for fog formation
    """
    # Wind factor reduces as wind speed increases. If WS is below the threshold, set the factor to 1.
    return pl.when(WS < threshold).then(1.0).otherwise(1 / WS)


def estimate_fog_intensity(T: pl.Expr, D: pl.Expr, WS: pl.Expr) -> pl.Expr:
    """
    Estimate the fog intensity based on temperature, dewpoint, and wind speed.
    T: Temperature in Celsius
    D: Dewpoint in Celsius
    WS: Wind Speed in m/s
    """
    RH = calculate_relative_humidity(T, D)
    WF = wind_factor(WS)

    # Assuming fog intensity is higher with higher relative humidity and lower wind speeds
    # This is a simple model and might need calibration with empirical data
    fog_intensity = RH * WF / 100  # Normalize to a 0-100 scale

    return fog_intensity


class FeaturesGenerator:
    def __init__(self, data_storage: DataStorage, cfg: config.ConfigPrepareData):
        self.data_storage = data_storage
        self.estonian_holidays = list(
            holidays.country_holidays("EE", years=range(2021, 2026)).keys()
        )
        self.cfg = cfg

    def _add_general_features(self, df_features):
        df_features = (
            df_features.with_columns(
                pl.col("datetime").dt.ordinal_day().alias("dayofyear"),
                pl.col("datetime").dt.hour().alias("hour"),
                pl.col("datetime").dt.day().alias("day"),
                pl.col("datetime").dt.weekday().alias("weekday"),
                pl.col("datetime").dt.month().alias("month"),
                pl.col("datetime").dt.year().alias("year"),
            )
            .with_columns(
                pl.concat_str(
                    "county",
                    "is_business",
                    "product_type",
                    "is_consumption",
                    separator="_",
                ).alias("segment"),
            )
            .with_columns(
                (np.pi * pl.col("dayofyear") / 183).sin().alias("sin(dayofyear)"),
                (np.pi * pl.col("dayofyear") / 183).cos().alias("cos(dayofyear)"),
                (np.pi * pl.col("hour") / 12).sin().alias("sin(hour)"),
                (np.pi * pl.col("hour") / 12).cos().alias("cos(hour)"),
            )
        )
        return df_features

    def _add_client_features(self, df_features):
        df_client = self.data_storage.df_client

        df_features = df_features.join(
            df_client.with_columns(
                (pl.col("date") + pl.duration(days=2)).cast(pl.Date)
            ),
            on=["county", "is_business", "product_type", "date"],
            how="left",
        ).with_columns(
            pl.col("installed_capacity").fill_null(0.0),
            pl.col("eic_count").fill_null(0.0),
        )

        return df_features

    def is_country_holiday(self, row):
        return (
            datetime.date(row["year"], row["month"], row["day"])
            in self.estonian_holidays
        )

    def _add_holidays_features(self, df_features):
        df_features = df_features.with_columns(
            pl.struct(["year", "month", "day"])
            .apply(self.is_country_holiday)
            .alias("is_country_holiday")
        )
        return df_features

    def _join_weather_with_counties(
        self,
        df_weather: pl.DataFrame,
        weather_columns: list[str],
    ):
        return (
            df_weather.with_columns(
                pl.col("latitude").cast(pl.datatypes.Float32),
                pl.col("longitude").cast(pl.datatypes.Float32),
            )
            .join(
                self.data_storage.df_stations,
                on=["longitude", "latitude"],
                how="left",
            )
            .filter(pl.col("county").is_not_null())
            .group_by(["datetime", "county"])
            .agg(*[(pl.col(c) * pl.col("weights")).sum() for c in weather_columns])
        )

    def _add_historical_weather_features(self, df_features: pl.DataFrame):
        lags = [
            0,
            1 * 24,
            2 * 24,
            3 * 24,
            4 * 24,
            5 * 24,
            6 * 24,
            7 * 24,
            8 * 24,
            9 * 24,
            10 * 24,
            11 * 24,
            12 * 24,
            13 * 24,
            14 * 24,
            15 * 24,
        ]

        datetime_start = df_features["datetime"].min() - timedelta(hours=max(lags))

        df_historical_weather = self.data_storage.df_historical_weather.filter(
            pl.col("datetime") >= datetime_start
        )

        df_historical_weather = (
            df_historical_weather.with_columns(
                pl.col("latitude").cast(pl.datatypes.Float32),
                pl.col("longitude").cast(pl.datatypes.Float32),
                (
                    pl.col("windspeed_10m")
                    * (pl.col("winddirection_10m") * 180 / np.pi).cos()
                ).alias("10_metre_u_wind_component"),
                (
                    pl.col("windspeed_10m")
                    * (pl.col("winddirection_10m") * 180 / np.pi).sin()
                ).alias("10_metre_v_wind_component"),
                estimate_fog_intensity(
                    T=pl.col("temperature"),
                    D=pl.col("dewpoint"),
                    WS=pl.col("windspeed_10m"),
                ).alias("fog"),
                calculate_relative_humidity(
                    T=pl.col("temperature"), D=pl.col("dewpoint")
                ).alias("humidity"),
            )
            .drop(columns=["winddirection_10m"])
        )

        weather_columns = self.data_storage.historical_weather_raw_features.copy()
        weather_columns.remove("winddirection_10m")
        weather_columns.extend(
            [
                "10_metre_u_wind_component",
                "10_metre_v_wind_component",
                "fog",
                "humidity",
            ]
        )

        df_historical_weather = self._join_weather_with_counties(
            df_historical_weather, weather_columns=weather_columns
        )

        integrated_lags = [24, 48, 72]

        df_historical_integrated_weather = (
            df_historical_weather
            .group_by("county")
            .map_groups(
                function=lambda df_g: df_g.with_columns(
                    *[
                        pl.col("rain").round(1).rolling_sum(lag, min_periods=1).alias(f"rain_integrated_{lag}h_historical")
                        for lag in integrated_lags
                    ],
                    *[
                        pl.col("snowfall").round(1).rolling_sum(lag, min_periods=1).alias(f"snowfall_integrated_{lag}h_historical")
                        for lag in integrated_lags
                    ],
                ) 
            )
            .with_columns(pl.col("datetime") + pl.duration(hours=1))
            .select(
                *[f"rain_integrated_{lag}h_historical" for lag in integrated_lags],
                *[f"snowfall_integrated_{lag}h_historical" for lag in integrated_lags],
                "county",
                "datetime"
            )
        )

        df_historical_weather = df_historical_weather.rename(
            {c: f"{c}_historical" for c in weather_columns}
        )

        for hours_lag in lags:
            df_features = df_features.join(
                df_historical_weather.with_columns(
                    pl.col("datetime") + pl.duration(hours=hours_lag)
                ),
                on=["datetime", "county"],
                how="left",
                suffix=f"_h{hours_lag}",
            ).join(
                df_historical_integrated_weather.with_columns(
                    pl.col("datetime") + pl.duration(hours=hours_lag)
                ),
                on=["datetime", "county"],
                how="left",
                suffix=f"_h{hours_lag}",
            )

        return df_features

    def _snowfall_mwe_to_cm(self, s: pl.Expr, k=424.24) -> pl.Expr:
        return s * k

    def _add_forecast_weather_features(self, df_features: pl.DataFrame):
        df_forecast_weather = self.data_storage.df_forecast_weather

        lags = [
            0,
            1 * 24,
            2 * 24,
            3 * 24,
            4 * 24,
            5 * 24,
            6 * 24,
            7 * 24,
            8 * 24,
            9 * 24,
            10 * 24,
            11 * 24,
            12 * 24,
            13 * 24,
            14 * 24,
            15 * 24,
        ]

        datetime_start = df_features["datetime"].min() - timedelta(max(lags))

        df_forecast_weather = (
            df_forecast_weather.filter((pl.col("hours_ahead") >= 23))
            .with_columns(
                (
                    pl.col("origin_datetime") + pl.duration(hours=pl.col("hours_ahead"))
                ).alias("datetime")
            )
            .drop(["hours_ahead", "forecast_datetime", "origin_datetime"])
            .filter(pl.col("datetime") >= datetime_start)
            .with_columns(
                pl.col("latitude").cast(pl.datatypes.Float32),
                pl.col("longitude").cast(pl.datatypes.Float32),
                (pl.col("total_precipitation") - pl.col("snowfall"))
                .clip(0.0)
                .alias("rain")
                * 1000,  # rain in mm
                self._snowfall_mwe_to_cm(pl.col("snowfall")),
                (
                    (
                        pl.col("10_metre_v_wind_component") ** 2
                        + pl.col("10_metre_u_wind_component") ** 2
                    )
                    ** (1 / 2)
                ).alias("windspeed_10m"),
            )
            .with_columns(
                estimate_fog_intensity(
                    T=pl.col("temperature"),
                    D=pl.col("dewpoint"),
                    WS=pl.col("windspeed_10m"),
                ).alias("fog"),
                calculate_relative_humidity(
                    T=pl.col("temperature"), D=pl.col("dewpoint")
                ).alias("humidity"),
            )
        )

        weather_columns = self.data_storage.forecast_weather_raw_features.copy()
        weather_columns.extend(["windspeed_10m", "rain", "fog", "humidity"])

        df_forecast_weather = self._join_weather_with_counties(
            df_forecast_weather, weather_columns=weather_columns
        )

        integrated_lags = [24, 48, 72]

        df_forecast_integrated_weather = (
            df_forecast_weather.with_columns(
                *[
                    pl.col("rain").clip(0.0).round(1).rolling_sum(window_size=lag, min_periods=1).over("county").alias(f"rain_integrated_{lag}h_forecast")
                    for lag in integrated_lags
                ],
                *[
                    pl.col("snowfall").clip(0.0).round(1).rolling_sum(window_size=lag, min_periods=1).over("county").alias(f"snowfall_integrated_{lag}h_forecast")
                    for lag in integrated_lags
                ],
                pl.col("datetime") + pl.duration(hours=1)
            ).select(
                *[f"rain_integrated_{lag}h_forecast" for lag in integrated_lags],
                *[f"snowfall_integrated_{lag}h_forecast" for lag in integrated_lags],
                "county",
                "datetime"
            )
        )

        df_forecast_weather = df_forecast_weather.rename(
            {c: f"{c}_forecast" for c in weather_columns}
        )

        for hours_lag in lags:
            df_features = df_features.join(
                df_forecast_weather.with_columns(
                    pl.col("datetime") + pl.duration(hours=hours_lag)
                ),
                on=["datetime", "county"],
                how="left",
                suffix=f"_h{hours_lag}",
            ).join(
                df_forecast_integrated_weather.with_columns(
                    pl.col("datetime") + pl.duration(hours=hours_lag)
                ),
                on=["datetime", "county"],
                how="left",
                suffix=f"_h{hours_lag}",
            )

        return df_features

    def _add_target_features(self, df_features):
        df_target = self.data_storage.df_target

        df_target_all_type_sum = (
            df_target.group_by(["datetime", "county", "is_business", "is_consumption"])
            .sum()
            .drop("product_type")
        )

        df_target_all_county_type_sum = (
            df_target.group_by(["datetime", "is_business", "is_consumption"])
            .sum()
            .drop("product_type", "county")
        )

        df_target_all_type_mean = (
            df_target.group_by(["datetime", "county", "is_business", "is_consumption"])
            .mean()
            .drop("product_type")
        )

        df_target_all_county_type_mean = (
            df_target.group_by(["datetime", "is_business", "is_consumption"])
            .mean()
            .drop("product_type", "county")
        )

        for hours_lag in [
            2 * 24,
            3 * 24,
            4 * 24,
            5 * 24,
            6 * 24,
            7 * 24,
            8 * 24,
            9 * 24,
            10 * 24,
            11 * 24,
            12 * 24,
            13 * 24,
            14 * 24,
        ]:
            df_features = (
                df_features.join(
                    df_target.with_columns(
                        pl.col("datetime") + pl.duration(hours=hours_lag)
                    ).rename({"target": f"target_{hours_lag}h"}),
                    on=[
                        "county",
                        "is_business",
                        "product_type",
                        "is_consumption",
                        "datetime",
                    ],
                    how="left",
                )
                .join(
                    df_target_all_type_mean.with_columns(
                        pl.col("datetime") + pl.duration(hours=hours_lag)
                    ).rename(
                        {
                            "target": f"target_all_type_mean_{hours_lag}h",
                        }
                    ),
                    on=["county", "is_business", "is_consumption", "datetime"],
                    how="left",
                )
                .join(
                    df_target_all_county_type_mean.with_columns(
                        pl.col("datetime") + pl.duration(hours=hours_lag)
                    ).rename(
                        {
                            "target": f"target_all_county_type_mean_{hours_lag}h",
                        }
                    ),
                    on=["is_business", "is_consumption", "datetime"],
                    how="left",
                )
                .with_columns(
                    pl.when(pl.col(f"target_{hours_lag}h").is_null())
                    .then(pl.col(f"target_all_type_mean_{hours_lag}h"))
                    .otherwise(pl.col(f"target_{hours_lag}h"))
                    .alias(f"target_{hours_lag}h"),
                )
                .with_columns(
                    pl.when(pl.col(f"target_{hours_lag}h").is_null())
                    .then(pl.col(f"target_all_county_type_mean_{hours_lag}h"))
                    .otherwise(pl.col(f"target_{hours_lag}h"))
                    .alias(f"target_{hours_lag}h"),
                )
            )

        for hours_lag in [2 * 24, 3 * 24, 7 * 24, 14 * 24]:
            df_features = df_features.join(
                df_target_all_type_sum.with_columns(
                    pl.col("datetime") + pl.duration(hours=hours_lag)
                ).rename({"target": f"target_all_type_sum_{hours_lag}h"}),
                on=["county", "is_business", "is_consumption", "datetime"],
                how="left",
            )

            df_features = df_features.join(
                df_target_all_county_type_sum.with_columns(
                    pl.col("datetime") + pl.duration(hours=hours_lag)
                ).rename({"target": f"target_all_county_type_sum_{hours_lag}h"}),
                on=["is_business", "is_consumption", "datetime"],
                how="left",
                suffix=f"_all_county_type_sum_{hours_lag}h",
            )

        cols_for_stats = [
            f"target_{hours_lag}h" for hours_lag in [2 * 24, 3 * 24, 4 * 24, 5 * 24]
        ]
        df_features = df_features.with_columns(
            df_features.select(cols_for_stats).mean(axis=1).alias(f"target_mean"),
            df_features.select(cols_for_stats)
            .transpose()
            .std()
            .transpose()
            .to_series()
            .alias(f"target_std"),
        )

        for target_prefix, lag_nominator, lag_denomonator in [
            ("target", 24 * 7, 24 * 14),
            ("target", 24 * 2, 24 * 9),
            ("target", 24 * 3, 24 * 10),
            ("target", 24 * 2, 24 * 3),
            ("target_all_type_sum", 24 * 2, 24 * 3),
            ("target_all_type_sum", 24 * 7, 24 * 14),
            ("target_all_county_type_sum", 24 * 2, 24 * 3),
            ("target_all_county_type_sum", 24 * 7, 24 * 14),
        ]:
            df_features = df_features.with_columns(
                (
                    pl.col(f"{target_prefix}_{lag_nominator}h")
                    / (pl.col(f"{target_prefix}_{lag_denomonator}h") + 1e-3)
                ).alias(f"{target_prefix}_ratio_{lag_nominator}_{lag_denomonator}")
            )

        return df_features

    def _add_target_norm_features(self, df_features):
        datetime_start = df_features["datetime"].min() - timedelta(hours=21 * 24)

        df_target = (
            self.data_storage.df_target.filter(pl.col("datetime") >= datetime_start)
            .with_columns(pl.col("datetime").cast(pl.Date).alias("date"))
            .join(
                self.data_storage.df_client.with_columns(pl.col("date").cast(pl.Date)),
                on=["county", "is_business", "product_type", "date"],
                how="left",
            )
            .with_columns(
                pl.col("installed_capacity").fill_null(0.0),
                pl.col("eic_count").fill_null(0.0),
            )
            .with_columns(
                pl.when(pl.col("eic_count") > 0)
                .then((pl.col("target") / pl.col("eic_count")))
                .otherwise(pl.col("target"))
                .alias("target_per_eic"),
                pl.when(pl.col("installed_capacity") > 0)
                .then((pl.col("target") / pl.col("installed_capacity")))
                .otherwise(pl.col("target"))
                .alias("target_per_capacity"),
            )
            .select(
                "datetime",
                "county",
                "is_business",
                "product_type",
                "is_consumption",
                "target_per_eic",
                "target_per_capacity",
            )
        )

        df_target_all_type_sum = (
            df_target.group_by(["datetime", "county", "is_business", "is_consumption"])
            .sum()
            .drop("product_type")
        )

        df_target_all_county_type_sum = (
            df_target.group_by(["datetime", "is_business", "is_consumption"])
            .sum()
            .drop("product_type", "county")
        )

        df_target_all_type_mean = (
            df_target.group_by(["datetime", "county", "is_business", "is_consumption"])
            .mean()
            .drop("product_type")
        )

        df_target_all_county_type_mean = (
            df_target.group_by(["datetime", "is_business", "is_consumption"])
            .mean()
            .drop("product_type", "county")
        )

        for hours_lag in [
            2 * 24,
            3 * 24,
            4 * 24,
            5 * 24,
            6 * 24,
            7 * 24,
            8 * 24,
            9 * 24,
            10 * 24,
            11 * 24,
            12 * 24,
            13 * 24,
            14 * 24,
            15 * 24,
        ]:
            df_features = (
                df_features.join(
                    df_target.with_columns(
                        pl.col("datetime") + pl.duration(hours=hours_lag)
                    ).rename(
                        {
                            "target_per_eic": f"target_per_eic_{hours_lag}h",
                            "target_per_capacity": f"target_per_capacity_{hours_lag}h",
                        }
                    ),
                    on=[
                        "county",
                        "is_business",
                        "product_type",
                        "is_consumption",
                        "datetime",
                    ],
                    how="left",
                )
                .join(
                    df_target_all_type_mean.with_columns(
                        pl.col("datetime") + pl.duration(hours=hours_lag)
                    ).rename(
                        {
                            "target_per_eic": f"target_per_eic_all_type_mean_{hours_lag}h",
                            "target_per_capacity": f"target_per_capacity_all_type_mean_{hours_lag}h",
                        }
                    ),
                    on=["county", "is_business", "is_consumption", "datetime"],
                    how="left",
                )
                .join(
                    df_target_all_county_type_mean.with_columns(
                        pl.col("datetime") + pl.duration(hours=hours_lag)
                    ).rename(
                        {
                            "target_per_eic": f"target_per_eic_all_county_type_mean_{hours_lag}h",
                            "target_per_capacity": f"target_per_capacity_all_county_type_mean_{hours_lag}h",
                        }
                    ),
                    on=["datetime", "is_business", "is_consumption"],
                    how="left",
                )
                .with_columns(
                    pl.when(pl.col(f"target_per_eic_{hours_lag}h").is_null())
                    .then(pl.col(f"target_per_eic_all_type_mean_{hours_lag}h"))
                    .otherwise(pl.col(f"target_per_eic_{hours_lag}h"))
                    .alias(f"target_per_eic_{hours_lag}h"),
                    pl.when(pl.col(f"target_per_capacity_{hours_lag}h").is_null())
                    .then(pl.col(f"target_per_capacity_all_type_mean_{hours_lag}h"))
                    .otherwise(pl.col(f"target_per_capacity_{hours_lag}h"))
                    .alias(f"target_per_capacity_{hours_lag}h"),
                )
                .with_columns(
                    pl.when(pl.col(f"target_per_eic_{hours_lag}h").is_null())
                    .then(pl.col(f"target_per_eic_all_county_type_mean_{hours_lag}h"))
                    .otherwise(pl.col(f"target_per_eic_{hours_lag}h"))
                    .alias(f"target_per_eic_{hours_lag}h"),
                    pl.when(pl.col(f"target_per_capacity_{hours_lag}h").is_null())
                    .then(
                        pl.col(f"target_per_capacity_all_county_type_mean_{hours_lag}h")
                    )
                    .otherwise(pl.col(f"target_per_capacity_{hours_lag}h"))
                    .alias(f"target_per_capacity_{hours_lag}h"),
                )
            )

        for hours_lag in [2 * 24, 3 * 24, 7 * 24, 14 * 24]:
            df_features = df_features.join(
                df_target_all_type_sum.with_columns(
                    pl.col("datetime") + pl.duration(hours=hours_lag)
                ).rename(
                    {
                        "target_per_eic": f"target_per_eic_all_type_sum_{hours_lag}h",
                        "target_per_capacity": f"target_per_capacity_all_type_sum_{hours_lag}h",
                    }
                ),
                on=["county", "is_business", "is_consumption", "datetime"],
                how="left",
            )

            df_features = df_features.join(
                df_target_all_county_type_sum.with_columns(
                    pl.col("datetime") + pl.duration(hours=hours_lag)
                ).rename(
                    {
                        "target_per_eic": f"target_per_eic_all_county_type_sum_{hours_lag}h",
                        "target_per_capacity": f"target_per_capacity_all_county_type_sum_{hours_lag}h",
                    }
                ),
                on=["is_business", "is_consumption", "datetime"],
                how="left",
                suffix=f"_all_county_type_sum_{hours_lag}h",
            )

        for norm_name in ["eic", "capacity"]:
            cols_for_stats = [
                f"target_per_{norm_name}_{hours_lag}h"
                for hours_lag in [2 * 24, 3 * 24, 4 * 24, 5 * 24]
            ]
            df_features = df_features.with_columns(
                df_features.select(cols_for_stats)
                .mean(axis=1)
                .alias(f"target_per_{norm_name}_mean"),
                df_features.select(cols_for_stats)
                .transpose()
                .std()
                .transpose()
                .to_series()
                .alias(f"target_per_{norm_name}_std"),
            )

            for target_prefix, lag_nominator, lag_denomonator in [
                (f"target_per_{norm_name}", 24 * 7, 24 * 14),
                (f"target_per_{norm_name}", 24 * 2, 24 * 9),
                (f"target_per_{norm_name}", 24 * 3, 24 * 10),
                (f"target_per_{norm_name}", 24 * 2, 24 * 3),
                (f"target_per_{norm_name}_all_type_sum", 24 * 2, 24 * 3),
                (f"target_per_{norm_name}_all_type_sum", 24 * 7, 24 * 14),
                (f"target_per_{norm_name}_all_county_type_sum", 24 * 2, 24 * 3),
                (f"target_per_{norm_name}_all_county_type_sum", 24 * 7, 24 * 14),
            ]:
                df_features = df_features.with_columns(
                    (
                        pl.col(f"{target_prefix}_{lag_nominator}h")
                        / (pl.col(f"{target_prefix}_{lag_denomonator}h") + 1e-3)
                    ).alias(f"{target_prefix}_ratio_{lag_nominator}_{lag_denomonator}")
                )

        return df_features

    def _add_target_norm_diff_features(self, df_features: pl.DataFrame) -> pl.DataFrame:
        diffs = [48, 168]
        lags = [d * 24 for d in (2, 3, 4, 5, 6, 7)]

        df_features = df_features.with_columns(
            *[
                (
                    pl.col(f"target_per_eic_{lag}h")
                    - pl.col(f"target_per_eic_{lag+diff}h")
                ).alias(f"target_per_eic_diff_{diff}h_lag{lag}h")
                for lag in lags
                for diff in diffs
            ]
        )

        for diff in diffs:
            cols_for_stats = [f"target_per_eic_diff_{diff}h_lag{lag}h" for lag in lags]
            df_features = df_features.with_columns(
                df_features.select(cols_for_stats)
                .transpose()
                .median()
                .transpose()
                .to_series()
                .alias(f"target_per_eic_diff_{diff}h_median"),
                df_features.select(cols_for_stats)
                .transpose()
                .quantile(0.25)
                .transpose()
                .to_series()
                .alias(f"target_per_eic_diff_{diff}h_q25"),
                df_features.select(cols_for_stats)
                .transpose()
                .quantile(0.75)
                .transpose()
                .to_series()
                .alias(f"target_per_eic_diff_{diff}h_q75"),
            )

        return df_features

    def _add_weather_diff_features(self, df_features: pl.DataFrame) -> pl.DataFrame:
        weather_features_both = [
            "temperature",
            "dewpoint",
            "rain",
            "snowfall",
            "cloudcover_total",
            "cloudcover_low",
            "cloudcover_mid",
            "cloudcover_high",
            "windspeed_10m",
            "10_metre_u_wind_component",
            "10_metre_v_wind_component",
            "humidity",
            "fog",
        ]
        diffs = [48, 168]
        suffixes = ["historical", "forecast"]
        for diff in diffs:
            df_features = df_features.with_columns(
                *[
                    (
                        pl.col(f"{feature}_{suffix}")
                        - pl.col(f"{feature}_historical_h{diff}")
                    ).alias(f"diff_{diff}h_{feature}_{suffix}")
                    for feature in weather_features_both
                    for suffix in suffixes
                ],
                (
                    pl.col("surface_solar_radiation_downwards_forecast")
                    - pl.col(f"surface_solar_radiation_downwards_forecast_h{diff}")
                ).alias(f"diff_{diff}h_surface_solar_radiation_downwards_forecast"),
            )
        return df_features

    def _reduce_memory_usage(self, df_features):
        df_features = df_features.with_columns(pl.col(pl.Float64).cast(pl.Float32))
        return df_features

    def _drop_columns(self, df_features):
        # df_features = df_features.drop("date", "hour", "dayofyear")
        return df_features

    def _to_pandas(self, df_features, y):
        cat_cols = [
            "county",
            "is_business",
            "product_type",
            "is_consumption",
            "segment",
        ]

        if y is not None:
            df_features = pd.concat([df_features.to_pandas(), y.to_pandas()], axis=1)
        else:
            df_features = df_features.to_pandas()

        df_features = df_features.set_index("row_id")
        df_features[cat_cols] = df_features[cat_cols].astype("category")

        return df_features

    def generate_features(self, df_prediction_items):
        if "target" in df_prediction_items.columns:
            df_prediction_items, y = (
                df_prediction_items.drop("target"),
                df_prediction_items.select("target"),
            )
        else:
            y = None

        df_features = df_prediction_items.with_columns(
            pl.col("datetime").cast(pl.Date).alias("date"),
        )

        for add_features in [
            self._add_general_features,
            self._add_client_features,
            self._add_forecast_weather_features,
            self._add_historical_weather_features,
            self._add_target_features,
            self._add_target_norm_features,
            self._add_holidays_features,
            self._add_target_norm_diff_features,
            self._add_weather_diff_features,
            self._reduce_memory_usage,
            self._drop_columns,
        ]:
            df_features = add_features(df_features)

        df_features = self._to_pandas(df_features, y)

        return df_features
