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

class FeaturesGenerator:
    def __init__(
            self, 
            data_storage: DataStorage,
            cfg: config.ConfigPrepareData
        ):
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

    def _add_forecast_weather_features(self, df_features):
        df_forecast_weather = self.data_storage.df_forecast_weather
        df_weather_station_to_county_mapping = (
            self.data_storage.df_weather_station_to_county_mapping
        )

        df_forecast_weather = (
            df_forecast_weather.rename({"forecast_datetime": "datetime"})
            .filter((pl.col("hours_ahead") >= 22) & pl.col("hours_ahead") <= 45)
            .drop("hours_ahead")
            .with_columns(
                pl.col("latitude").cast(pl.datatypes.Float32),
                pl.col("longitude").cast(pl.datatypes.Float32),
            )
            .join(
                df_weather_station_to_county_mapping,
                how="left",
                on=["longitude", "latitude"],
            )
            .drop("longitude", "latitude")
        )

        df_forecast_weather_date = (
            df_forecast_weather.group_by("datetime").mean().drop("county")
        )

        df_forecast_weather_local = (
            df_forecast_weather.filter(pl.col("county").is_not_null())
            .group_by("county", "datetime")
            .mean()
        )

        for hours_lag in [0, 7 * 24]:
            df_features = df_features.join(
                df_forecast_weather_date.with_columns(
                    pl.col("datetime") + pl.duration(hours=hours_lag)
                ),
                on="datetime",
                how="left",
                suffix=f"_forecast_{hours_lag}h",
            )
            df_features = df_features.join(
                df_forecast_weather_local.with_columns(
                    pl.col("datetime") + pl.duration(hours=hours_lag)
                ),
                on=["county", "datetime"],
                how="left",
                suffix=f"_forecast_local_{hours_lag}h",
            )

        return df_features

    def _join_weather_with_counties(self, df_weather: pl.DataFrame):
        
        weather_columns = [
            "temperature",
            "dewpoint",
            "rain",
            "snowfall",
            "surface_pressure",
            "cloudcover_total",
            "cloudcover_low",
            "cloudcover_mid",
            "cloudcover_high",
            "windspeed_10m",
            "winddirection_10m",
            "shortwave_radiation",
            "direct_solar_radiation",
            "diffuse_radiation",
        ]
        
        def calculate_weights_of_stations(df: pl.DataFrame, min_weight=0.3):
            d_max = df["dist"].max()
            d_ref = d_max / min_weight if min_weight > 0 else d_max

            df = (
                df
                .with_columns(
                    (pl.col("dist") / d_ref).clip(lower_bound=0.0, upper_bound=None).alias("weights")
                )
                .with_columns(
                    (pl.col("weights") / pl.col("weights").sum()).alias("weights")
                )
            )

            return df
        
        def find_relevant_county_stations(
            df_capitals: pl.DataFrame, 
            df_weather_hist: pl.DataFrame,
            max_dist: float = 100.0,
            on_land_only: bool = True,
        ):
            # Creating unique weather stations
            weather_stations = df_weather_hist.select(["longitude", "latitude"]).unique().to_numpy()
            weather_stations = [Point(lon, lat) for lon, lat in weather_stations]

            # Function to calculate distances and filter based on conditions
            def filter_stations(capital_row):
                lat, lon, county = capital_row["latitude"], capital_row["longitude"], capital_row["county"]
                capital_loc = Point(lon, lat)
                data = []
                for station in weather_stations:
                    dist = geopy.distance.geodesic((capital_loc.x, capital_loc.y), (station.x, station.y)).km
                    if dist < max_dist:
                        if on_land_only and not globe.is_land(lon=station.x, lat=station.y):
                            continue
                        data.append((county, station.x, station.y, dist))
                return data

            # Apply the function to each row of df_capitals and flatten the result
            result_rows = list(chain.from_iterable(filter_stations(row) for row in df_capitals.rows(named=True)))
                
            return pl.DataFrame(result_rows, schema={"county": pl.Int64, "longitude": pl.Float32, "latitude": pl.Float32, "dist": pl.Float64})

        def average_weather_based_on_weights(df: pl.DataFrame):
            return df.select(
                pl.col("datetime").first(),
                pl.col("county").first(),
                *[
                    pl.col(c).dot(pl.col("weights"))
                    for c in weather_columns
                ]
            )

        df_relevant_stations = find_relevant_county_stations(
            df_capitals=self.data_storage.df_capitals,
            df_weather_hist=df_weather,
            max_dist=100.0,
            on_land_only=True
        )
        df_stations_with_weights = df_relevant_stations.group_by("county").map_groups(calculate_weights_of_stations)
        
        return (
            df_weather
            .with_columns(
                pl.col("latitude").cast(pl.datatypes.Float32),
                pl.col("longitude").cast(pl.datatypes.Float32),
            )
            .join(df_stations_with_weights, on=["longitude", "latitude"], how="left",)
            .group_by(["datetime", "county"]).map_groups(average_weather_based_on_weights)
            .filter(pl.col("county").is_not_null())
        )


    def _add_historical_weather_features(self, df_features):
        df_historical_weather = self.data_storage.df_historical_weather

        df_historical_weather = (
            df_historical_weather.with_columns(
                pl.col("latitude").cast(pl.datatypes.Float32),
                pl.col("longitude").cast(pl.datatypes.Float32),
            )
        )

        df_historical_weather = self._join_weather_with_counties(df_historical_weather)
        
        df_features = df_features.join(
            df_historical_weather,
            on=["datetime", "county"],
            how="left"
        )

        for hours_lag in [1*24, 2 * 24, 7 * 24]:
            df_features = df_features.join(
                df_historical_weather.with_columns(
                    pl.col("datetime") + pl.duration(hours=hours_lag)
                ),
                on=["datetime", "county"],
                how="left",
                suffix=f"_historical_local_{hours_lag}h",
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
            df_features = df_features.join(
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

    def _reduce_memory_usage(self, df_features):
        df_features = df_features.with_columns(pl.col(pl.Float64).cast(pl.Float32))
        return df_features

    def _drop_columns(self, df_features):
        df_features = df_features.drop("date", "hour", "dayofyear")
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
            #self._add_forecast_weather_features,
            self._add_historical_weather_features,
            self._add_target_features,
            self._add_holidays_features,
            self._reduce_memory_usage,
            self._drop_columns,
        ]:
            df_features = add_features(df_features)

        df_features = self._to_pandas(df_features, y)

        return df_features


if __name__ == '__main__':
    ds = DataStorage(
        Path("/beegfs/ws/0/s4610340-energy_behavior/yahor/kaggle-predict_energy_behavior_of_prosumers/data/raw"),
        Path("/beegfs/ws/0/s4610340-energy_behavior/yahor/kaggle-predict_energy_behavior_of_prosumers/data/processed/download_geo_data"),
    )
    df_features_train = FeaturesGenerator(ds, cfg={}).generate_features(ds.df_data)

    print(df_features_train)

    print(df_features_train.columns)