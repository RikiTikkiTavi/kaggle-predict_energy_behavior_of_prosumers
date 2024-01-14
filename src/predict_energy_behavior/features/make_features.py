from pathlib import Path
import select
from typing import Sequence
import polars as pl
import numpy as np
import predict_energy_behavior.data.read as read
import predict_energy_behavior.data.constants as data_const
import predict_energy_behavior.config as config
import predict_energy_behavior.features.process as process

from functools import partial

import hydra
import logging

import pandas as pd

_logger = logging.getLogger(__file__)

def average_weather_based_on_weights(df: pd.DataFrame, min_weight=0.3):
    weights = df["weights"]
    df = df.drop(["weights", "dist"], axis=1)
    return pd.Series(np.average(df.to_numpy(), axis=0, weights=weights), index=df.columns)


def make_train_features(
        datasets: read.RawDatasets,
        df_target: pl.DataFrame,
        df_capitals: pd.DataFrame,
        conf: config.ConfigPrepareData,
        lags_target: Sequence[int] = (1,7),
        stations_max_dist: float = 100,
        stations_on_land_only: bool = True
):
    df_data = datasets.data
    df_client = datasets.client
    df_gas_prices = datasets.gas
    df_electricity_prices = datasets.electricity
    df_historical_weather = datasets.weather_hist
    df_forecast_weather = datasets.weather_forecast

    county_name_to_id = {v.lower().capitalize(): int(k) for k, v in datasets.county_id_to_name.items()}

    _logger.info("Finding relevant weather stations ...")

    df_weather_stations: pd.DataFrame = pl.DataFrame(
        process.find_relevant_county_stations(
            df_capitals=df_capitals,
            df_weather_hist=df_historical_weather.to_pandas(),
            max_dist=stations_max_dist,
            on_land_only=stations_on_land_only
        )
        .groupby("county_name", as_index=False)
        .apply(process.calculate_weights_of_stations).reset_index(drop=True)
    ).with_columns(
        pl.col("latitude").cast(pl.datatypes.Float32), 
        pl.col("longitude").cast(pl.datatypes.Float32)
    )

    _logger.info("Processing historical weather ...")
    
    df_historical_weather = pl.from_pandas(
        df_historical_weather
        .with_columns(
            pl.col("latitude").cast(pl.datatypes.Float32), 
            pl.col("longitude").cast(pl.datatypes.Float32)
        )
        .join(df_weather_stations, how="left", on=["longitude", "latitude"])
        .drop("longitude", "latitude")
        .to_pandas()
        .groupby(["datetime", "county_name"])
        .apply(partial(average_weather_based_on_weights, min_weight=0.3))
        .reset_index()
        .assign(**{"county": lambda df: df["county_name"].map(county_name_to_id)})
        .drop(columns=["county_name"])
    ).with_columns(pl.col("county").cast(pl.Int64))

    _logger.info("Processing forecast weather ...")

    df_forecast_weather = pl.from_pandas(
        df_forecast_weather
        .rename({"forecast_datetime": "datetime"})
        .filter(pl.col("hours_ahead") >= 24) # we don't need forecast for today
        .with_columns(
            pl.col("latitude").cast(pl.datatypes.Float32), 
            pl.col("longitude").cast(pl.datatypes.Float32),
            # datetime for forecast in a different timezone
            pl.col('datetime').dt.replace_time_zone(None).cast(pl.Datetime("us")),
        )
        .join(df_weather_stations, how="left", on=["longitude", "latitude"])
        .drop("longitude", "latitude")
        .to_pandas()
        .groupby(["datetime", "county_name"])
        .apply(partial(average_weather_based_on_weights, min_weight=0.3))
        .reset_index()
        .assign(**{"county": lambda df: df["county_name"].map(county_name_to_id)})
        .drop(columns=["county_name"])
    ).with_columns(pl.col("county").cast(pl.Int64))

    df_data = (
        df_data
        .with_columns(pl.col("datetime").cast(pl.Date).alias("date"))
        .join(
            df_client
                .with_columns((pl.col("date") + pl.duration(days=2))
                .cast(pl.Date)), 
            on=["county", "is_business", "product_type", "date"], 
            how="left"
        )
    )
    
    df_gas_prices = (
        df_gas_prices.rename({"forecast_date": "date"})
    )
    
    df_electricity_prices = (
        df_electricity_prices.rename({"forecast_date": "datetime"})
    )
    
    df_data = (
        df_data
        # pl.duration(days=1) shifts datetime to join lag features (usually we join last available values)
        .join(df_gas_prices.with_columns((pl.col("date") + pl.duration(days=1)).cast(pl.Date)), on="date", how="left")
        .join(df_electricity_prices.with_columns(pl.col("datetime") + pl.duration(days=1)), on="datetime", how="left")
        
        # lag forecast_weather features (24 hours * days)
        .join(df_forecast_weather, on=["county", "datetime"], how="left", suffix="_fl")
        .join(df_forecast_weather.with_columns(pl.col("datetime") + pl.duration(days=7)), on=["county", "datetime"], how="left", suffix="_fl_7d")

        # lag historical_weather features (24 hours * days)
        .join(df_historical_weather.with_columns(pl.col("datetime")), on=["county", "datetime"], how="left")
        .join(df_historical_weather.with_columns(pl.col("datetime") + pl.duration(days=2)), on=["county", "datetime"], how="left", suffix="_hl_2d")
        .join(df_historical_weather.with_columns(pl.col("datetime") + pl.duration(days=7)), on=["county", "datetime"], how="left", suffix="_hl_7d")
    )

    # lag target features (24 hours * days)
    for target_lag in lags_target:
        df_data = df_data.join(
            df_target
                .with_columns(pl.col("datetime") + pl.duration(days=target_lag))
                .rename({"target": f"target_d{target_lag}"}), 
            on=["county", "is_business", "product_type", "is_consumption", "datetime"], 
            how="left"
        )
        
    df_data = (
        df_data
        .with_columns(
            pl.col("datetime").dt.ordinal_day().alias("dayofyear"),
            pl.col("datetime").dt.hour().alias("hour"),
            pl.col("datetime").dt.day().alias("day"),
            pl.col("datetime").dt.weekday().alias("weekday"),
            pl.col("datetime").dt.month().alias("month"),
            pl.col("datetime").dt.year().alias("year"),
        )
        .with_columns(
            pl.concat_str("county", "is_business", "product_type", "is_consumption", separator="_").alias("segment"),
        )
        # cyclical features encoding https://towardsdatascience.com/cyclical-features-encoding-its-about-time-ce23581845ca
        .with_columns(
            (np.pi * pl.col("dayofyear") / 183).sin().alias("sin(dayofyear)"),
            (np.pi * pl.col("dayofyear") / 183).cos().alias("cos(dayofyear)"),
            (np.pi * pl.col("hour") / 12).sin().alias("sin(hour)"),
            (np.pi * pl.col("hour") / 12).cos().alias("cos(hour)"),
        )
        .with_columns(
            pl.col(pl.Float64).cast(pl.Float32),
        )
        .drop("hour", "dayofyear")
        .drop_nulls(["target"])
    )
    
    return df_data

@hydra.main(config_path="../../../configs", config_name="prepare_data", version_base="1.2")
def prepare_data(cfg: config.ConfigPrepareData):
    _logger.info("Preparing data ...")
    path_data_processed = Path.cwd()
    path_data_raw = Path(cfg.dir.data_raw)

    _logger.info(f"Reading raw datasets from {path_data_raw} ...")
    datasets = read.read_datasets_from_folder(path_data_raw)

    _logger.info("Load capitals ...")
    df_capitals = read.load_capitals()

    _logger.info("Processing data ...")
    df_train_features = make_train_features(
        datasets=datasets,
        conf=cfg,
        df_target=datasets.data.select(data_const.target_cols),
        df_capitals=df_capitals
    )

    path_data_processed.mkdir(exist_ok=True, parents=True)
    path_df_train = path_data_processed / "df_train_features.parquet"
    _logger.info(f"Saving processed data to {path_df_train} ...")
    df_train_features.write_parquet(path_df_train)


if __name__=="__main__":
    prepare_data()