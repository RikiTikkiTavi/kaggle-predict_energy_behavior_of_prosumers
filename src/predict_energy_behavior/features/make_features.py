from pathlib import Path
import select
from typing import Sequence
import polars as pl
import numpy as np
import predict_energy_behavior.data.read as read
import predict_energy_behavior.data.constants as data_const
import predict_energy_behavior.config as config
import hydra
import logging

_logger = logging.getLogger(__file__)

def make_train_features(
        datasets: read.RawDatasets,
        df_target: pl.DataFrame,
        lags_target: Sequence[int] = (2,3,4,5,6,7,14),
):
    df_data = datasets.data
    df_client = datasets.client
    df_gas_prices = datasets.gas
    df_electricity_prices = datasets.electricity
    df_historical_weather = datasets.weather_hist
    df_forecast_weather = datasets.weather_forecast
    df_weather_station_to_county_mapping = datasets.weather_station_to_county

    df_data = (
        df_data.with_columns(pl.col("datetime").cast(pl.Date).alias("date"),)
    )
    
    df_gas_prices = (
        df_gas_prices.rename({"forecast_date": "date"})
    )
    
    df_electricity_prices = (
        df_electricity_prices.rename({"forecast_date": "datetime"})
    )
    
    df_weather_station_to_county_mapping = (
        df_weather_station_to_county_mapping.with_columns(pl.col("latitude").cast(pl.datatypes.Float32),pl.col("longitude").cast(pl.datatypes.Float32))
    )
    
    # sum of all product_type targets related to ["datetime", "county", "is_business", "is_consumption"]
    df_target_all_type_sum = (
        df_target.group_by(["datetime", "county", "is_business", "is_consumption"]).sum().drop("product_type")
    )
    
    df_forecast_weather = (
        df_forecast_weather.rename({"forecast_datetime": "datetime"}).filter(pl.col("hours_ahead") >= 24) # we don't need forecast for today
        .with_columns(pl.col("latitude").cast(pl.datatypes.Float32),pl.col("longitude").cast(pl.datatypes.Float32),
            # datetime for forecast in a different timezone
            pl.col('datetime').dt.replace_time_zone(None).cast(pl.Datetime("us")),
        )
        .join(df_weather_station_to_county_mapping, how="left", on=["longitude", "latitude"]).drop("longitude", "latitude")
    )
    
    df_historical_weather = (
        df_historical_weather
        .with_columns(pl.col("latitude").cast(pl.datatypes.Float32),pl.col("longitude").cast(pl.datatypes.Float32),
        )
        .join(df_weather_station_to_county_mapping, how="left", on=["longitude", "latitude"]).drop("longitude", "latitude")
    )
    
    # creating average forecast characteristics for all weather stations
    df_forecast_weather_date = (
        df_forecast_weather.group_by("datetime").mean().drop("county")
    )
    
    # creating average forecast characteristics for weather stations related to county
    df_forecast_weather_local = (
        df_forecast_weather.filter(pl.col("county").is_not_null()).group_by("county", "datetime").mean()
    )
    
    # creating average historical characteristics for all weather stations
    df_historical_weather_date = (
        df_historical_weather.group_by("datetime").mean().drop("county")
    )
    
    # creating average historical characteristics for weather stations related to county
    df_historical_weather_local = (
        df_historical_weather.filter(pl.col("county").is_not_null()).group_by("county", "datetime").mean()
    )
    
    df_data = (
        df_data
        # pl.duration(days=1) shifts datetime to join lag features (usually we join last available values)
        .join(df_gas_prices.with_columns((pl.col("date") + pl.duration(days=1)).cast(pl.Date)), on="date", how="left")
        .join(
            df_client
                .with_columns((pl.col("date") + pl.duration(days=2))
                .cast(pl.Date)), 
            on=["county", "is_business", "product_type", "date"], 
            how="left"
        )
        .join(df_electricity_prices.with_columns(pl.col("datetime") + pl.duration(days=1)), on="datetime", how="left")
        
        # lag forecast_weather features (24 hours * days)
        .join(df_forecast_weather_date, on="datetime", how="left", suffix="_fd")
        .join(df_forecast_weather_local, on=["county", "datetime"], how="left", suffix="_fl")
        .join(df_forecast_weather_date.with_columns(pl.col("datetime") + pl.duration(days=7)), on="datetime", how="left", suffix="_fd_7d")
        .join(df_forecast_weather_local.with_columns(pl.col("datetime") + pl.duration(days=7)), on=["county", "datetime"], how="left", suffix="_fl_7d")

        # lag historical_weather features (24 hours * days)
        .join(df_historical_weather_date.with_columns(pl.col("datetime") + pl.duration(days=2)), on="datetime", how="left", suffix="_hd_2d")
        .join(df_historical_weather_local.with_columns(pl.col("datetime") + pl.duration(days=2)), on=["county", "datetime"], how="left", suffix="_hl_2d")
        .join(df_historical_weather_date.with_columns(pl.col("datetime") + pl.duration(days=7)), on="datetime", how="left", suffix="_hd_7d")
        .join(df_historical_weather_local.with_columns(pl.col("datetime") + pl.duration(days=7)), on=["county", "datetime"], how="left", suffix="_hl_7d")
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
        .join(df_target_all_type_sum.with_columns(pl.col("datetime") + pl.duration(days=2)).rename({"target": "target_1"}), on=["county", "is_business", "is_consumption", "datetime"], suffix="_all_type_sum", how="left")
        .join(df_target_all_type_sum.with_columns(pl.col("datetime") + pl.duration(days=3)).rename({"target": "target_2"}), on=["county", "is_business", "is_consumption", "datetime"], suffix="_all_type_sum", how="left")
        .join(df_target_all_type_sum.with_columns(pl.col("datetime") + pl.duration(days=7)).rename({"target": "target_6"}), on=["county", "is_business", "is_consumption", "datetime"], suffix="_all_type_sum", how="left")
        .join(df_target_all_type_sum.with_columns(pl.col("datetime") + pl.duration(days=14)).rename({"target": "target_7"}), on=["county", "is_business", "is_consumption", "datetime"], suffix="_all_type_sum", how="left")
        
        
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
    
    _logger.info("Processing data ...")
    df_train_features = make_train_features(
        datasets=datasets,
        df_target=datasets.data.select(data_const.target_cols)
    )

    path_data_processed.mkdir(exist_ok=True, parents=True)
    path_df_train = path_data_processed / "df_train_features.parquet"
    _logger.info(f"Saving processed data to {path_df_train} ...")
    df_train_features.write_parquet(path_df_train)


if __name__=="__main__":
    prepare_data()