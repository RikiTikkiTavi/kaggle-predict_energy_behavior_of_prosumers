from datetime import datetime
import json
from pathlib import Path
import polars as pl
import os
import pandas as pd

from predict_energy_behavior.utils.common import format_county_name


class DataStorage:
    root: str

    data_cols = [
        "target",
        "county",
        "is_business",
        "product_type",
        "is_consumption",
        "datetime",
        "row_id",
    ]
    client_cols = [
        "product_type",
        "county",
        "eic_count",
        "installed_capacity",
        "is_business",
        "date",
    ]
    gas_prices_cols = ["forecast_date", "lowest_price_per_mwh", "highest_price_per_mwh"]
    electricity_prices_cols = ["forecast_date", "euros_per_mwh"]
    forecast_weather_cols = [
        "latitude",
        "longitude",
        "origin_datetime",
        "hours_ahead",
        "temperature",
        "dewpoint",
        "cloudcover_high",
        "cloudcover_low",
        "cloudcover_mid",
        "cloudcover_total",
        "10_metre_u_wind_component",
        "10_metre_v_wind_component",
        "forecast_datetime",
        "direct_solar_radiation",
        "surface_solar_radiation_downwards",
        "snowfall",
        "total_precipitation",
    ]
    forecast_weather_raw_features = [
        "temperature",
        "dewpoint",
        "cloudcover_high",
        "cloudcover_low",
        "cloudcover_mid",
        "cloudcover_total",
        "10_metre_u_wind_component",
        "10_metre_v_wind_component",
        "direct_solar_radiation",
        "surface_solar_radiation_downwards",
        "snowfall",
        "total_precipitation",
    ]

    historical_weather_cols = [
        "datetime",
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
        "latitude",
        "longitude",
    ]

    historical_weather_raw_features = [
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
    location_cols = ["longitude", "latitude", "county"]
    target_cols = [
        "target",
        "county",
        "is_business",
        "product_type",
        "is_consumption",
        "datetime",
    ]

    def _fill_dst_nulls(self, df_target: pl.DataFrame) -> pl.DataFrame:
        dfs = []
        for _, df in df_target.group_by(
            ["county", "is_business", "product_type", "is_consumption"],
            maintain_order=True,
        ):
            dfs.append(
                df.with_columns(
                    pl.col("target").fill_null(strategy="forward", limit=1)
                )
            )

        return pl.concat(dfs)


    def _fill_nulls_historical(self, df: pl.DataFrame) -> pl.DataFrame:
        df_prev_day = df.filter(
            (pl.col("datetime") >= datetime(year=2023, month=5, day=28, hour=11)) &
            (pl.col("datetime") <= datetime(year=2023, month=5, day=29, hour=23))
        ).with_columns(
            pl.col("datetime") + pl.duration(hours=48)
        )
        df = df.filter(
            (pl.col("datetime") < datetime(year=2023, month=5, day=30, hour=11))
        )

        return pl.concat([
            df,
            df_prev_day
        ])



    def __init__(
        self,
        path_data_raw: Path,
        path_data_geo: Path,
        path_data_stations: Path,
    ):
        self.root = str(path_data_raw)

        self.df_data = self._fill_dst_nulls(pl.read_csv(
            os.path.join(self.root, "train.csv"),
            columns=self.data_cols,
            try_parse_dates=True,
        )).filter(
            pl.col("datetime") >= datetime(year=2022, month=1, day=1)
        )

        self.df_client = pl.read_csv(
            os.path.join(self.root, "client.csv"),
            columns=self.client_cols,
            try_parse_dates=True,
        )

        self.df_gas_prices = pl.read_csv(
            os.path.join(self.root, "gas_prices.csv"),
            columns=self.gas_prices_cols,
            try_parse_dates=True,
        )
        self.df_electricity_prices = pl.read_csv(
            os.path.join(self.root, "electricity_prices.csv"),
            columns=self.electricity_prices_cols,
            try_parse_dates=True,
        )
        self.df_forecast_weather = pl.read_csv(
            os.path.join(self.root, "forecast_weather.csv"),
            columns=self.forecast_weather_cols,
            try_parse_dates=True,
        )
        self.df_historical_weather = self._fill_nulls_historical(pl.read_csv(
            os.path.join(self.root, "historical_weather.csv"),
            columns=self.historical_weather_cols,
            try_parse_dates=True,
        ))

        self.df_weather_station_to_county_mapping = pl.read_csv(
            os.path.join(self.root, "weather_station_to_county_mapping.csv"),
            columns=self.location_cols,
            try_parse_dates=True,
        )
        # self.df_data = self.df_data.filter(
        #    pl.col("datetime") >= pd.to_datetime("2022-01-01")
        # )
        self.df_target = self._fill_dst_nulls(self.df_data.select(self.target_cols))

        self.df_capitals = pl.read_csv(path_data_geo / "capitals.csv")
        self.df_county_boundaries = pl.read_csv(path_data_geo / "county_boundaries.csv")

        with open(path_data_raw / "county_id_to_name_map.json", "r") as file:
            county_id_to_name = json.load(file)
            county_name_to_id = {
                format_county_name(n): int(i) for i, n in county_id_to_name.items()
            }

        self.df_capitals = self.df_capitals.with_columns(
            pl.col("county_name").map_dict(county_name_to_id).alias("county")
        ).drop("county_name")
        self.df_county_boundaries = self.df_county_boundaries.with_columns(
            pl.col("county_name").map_dict(county_name_to_id).alias("county")
        ).drop("county_name")

        self.df_stations = pl.read_parquet(
            path_data_stations / "stations_with_weights.parquet"
        )

        self.schema_data = self.df_data.schema
        self.schema_client = self.df_client.schema
        self.schema_gas_prices = self.df_gas_prices.schema
        self.schema_electricity_prices = self.df_electricity_prices.schema
        self.schema_forecast_weather = self.df_forecast_weather.schema
        self.schema_historical_weather = self.df_historical_weather.schema
        self.schema_target = self.df_target.schema
        self.schema_county_boundaries = self.df_county_boundaries.schema
        self.schema_capitals = self.df_capitals.schema

        self.df_weather_station_to_county_mapping = (
            self.df_weather_station_to_county_mapping.with_columns(
                pl.col("latitude").cast(pl.datatypes.Float32),
                pl.col("longitude").cast(pl.datatypes.Float32),
            )
        )

    def update_with_new_data(
        self,
        df_new_client,
        df_new_gas_prices,
        df_new_electricity_prices,
        df_new_forecast_weather,
        df_new_historical_weather,
        df_new_target,
    ):
        df_new_client = pl.from_pandas(
            df_new_client[self.client_cols], schema_overrides=self.schema_client
        )
        df_new_gas_prices = pl.from_pandas(
            df_new_gas_prices[self.gas_prices_cols],
            schema_overrides=self.schema_gas_prices,
        )
        df_new_electricity_prices = pl.from_pandas(
            df_new_electricity_prices[self.electricity_prices_cols],
            schema_overrides=self.schema_electricity_prices,
        )
        df_new_forecast_weather = pl.from_pandas(
            df_new_forecast_weather[self.forecast_weather_cols],
            schema_overrides=self.schema_forecast_weather,
        )
        df_new_historical_weather = pl.from_pandas(
            df_new_historical_weather[self.historical_weather_cols],
            schema_overrides=self.schema_historical_weather,
        )
        df_new_target = self._fill_dst_nulls(
            pl.from_pandas(
                df_new_target[self.target_cols], schema_overrides=self.schema_target
            )
        )

        self.df_client = pl.concat([self.df_client, df_new_client]).unique(
            ["date", "county", "is_business", "product_type"]
        )
        self.df_gas_prices = pl.concat([self.df_gas_prices, df_new_gas_prices]).unique(
            ["forecast_date"]
        )
        self.df_electricity_prices = pl.concat(
            [self.df_electricity_prices, df_new_electricity_prices]
        ).unique(["forecast_date"])
        self.df_forecast_weather = pl.concat(
            [self.df_forecast_weather, df_new_forecast_weather]
        ).unique(["forecast_datetime", "latitude", "longitude", "hours_ahead"])
        self.df_historical_weather = pl.concat(
            [self.df_historical_weather, df_new_historical_weather]
        ).unique(["datetime", "latitude", "longitude"])
        self.df_target = pl.concat([self.df_target, df_new_target]).unique(
            ["datetime", "county", "is_business", "product_type", "is_consumption"]
        )

    def preprocess_test(self, df_test: pd.DataFrame):
        df_test = df_test.rename(columns={"prediction_datetime": "datetime"})
        df_test = pl.from_pandas(
            df_test[self.data_cols[1:]], schema_overrides=self.schema_data
        )
        return df_test
