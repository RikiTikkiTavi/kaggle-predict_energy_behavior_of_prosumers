from pathlib import Path
from typing import NamedTuple
import predict_energy_behavior.data.constants as constants
from dataclasses import dataclass
import polars as pl


@dataclass
class RawDatasets:
    data: pl.DataFrame
    client: pl.DataFrame
    gas: pl.DataFrame
    electricity: pl.DataFrame
    weather_forecast: pl.DataFrame
    weather_hist: pl.DataFrame
    weather_station_to_county: pl.DataFrame


def read_datasets_from_folder(path: Path) -> RawDatasets:
    return RawDatasets(
        data=pl.read_csv(path / "train.csv", columns=constants.data_cols, try_parse_dates=True),
        client=pl.read_csv(path / "client.csv", columns=constants.client_cols, try_parse_dates=True),
        gas=pl.read_csv(path / "gas_prices.csv", columns=constants.gas_prices_cols, try_parse_dates=True),
        electricity=pl.read_csv(path / "electricity_prices.csv", columns=constants.electricity_prices_cols, try_parse_dates=True),
        weather_forecast=pl.read_csv(path / "forecast_weather.csv", columns=constants.forecast_weather_cols, try_parse_dates=True),
        weather_hist= pl.read_csv(path / "historical_weather.csv", columns=constants.historical_weather_cols, try_parse_dates=True),
        weather_station_to_county=pl.read_csv(path / "weather_station_to_county_mapping.csv", columns=constants.location_cols, try_parse_dates=True)
    )