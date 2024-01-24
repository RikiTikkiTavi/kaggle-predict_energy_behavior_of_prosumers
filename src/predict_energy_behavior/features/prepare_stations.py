import datetime
from functools import partial
from itertools import chain
import json
import logging
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
import hydra

_logger = logging.getLogger(__name__)


def find_relevant_county_stations(
    df_capitals: pl.DataFrame,
    df_weather_hist: pl.DataFrame,
    max_dist: float = 100.0,
    on_land_only: bool = True,
):
    # Creating unique weather stations
    weather_stations = (
        df_weather_hist.select(["longitude", "latitude"]).unique().to_numpy()
    )
    weather_stations = [Point(lon, lat) for lon, lat in weather_stations]

    # Function to calculate distances and filter based on conditions
    def filter_stations(capital_row):
        lat, lon, county = (
            capital_row["latitude"],
            capital_row["longitude"],
            capital_row["county"],
        )
        capital_loc = Point(lon, lat)
        data = []
        for station in weather_stations:
            dist = geopy.distance.geodesic(
                (capital_loc.x, capital_loc.y), (station.x, station.y)
            ).km
            if dist < max_dist:
                if on_land_only and not globe.is_land(lon=station.x, lat=station.y):
                    continue
                data.append((county, station.x, station.y, dist))
        return data

    # Apply the function to each row of df_capitals and flatten the result
    result_rows = list(
        chain.from_iterable(
            filter_stations(row) for row in df_capitals.rows(named=True)
        )
    )

    return pl.DataFrame(
        result_rows,
        schema={
            "county": pl.Int64,
            "longitude": pl.Float32,
            "latitude": pl.Float32,
            "dist": pl.Float64,
        },
    )


def calculate_weights_of_stations(df: pl.DataFrame, min_weight=0.3):
    d_max = df["dist"].max()
    d_ref = d_max / min_weight if min_weight > 0 else d_max

    df = df.with_columns(
        (pl.col("dist") / d_ref)
        .clip(lower_bound=0.0, upper_bound=None)
        .alias("weights")
    ).with_columns((pl.col("weights") / pl.col("weights").sum()).alias("weights"))

    return df


@hydra.main(
    config_path="../../../configs", config_name="prepare_stations", version_base="1.3"
)
def prepare_stations(cfg: config.ConfigPrepareStations):
    _logger.info("Preparing data ...")
    path_data_processed = Path.cwd()
    path_data_raw = Path(cfg.dir.data_raw)

    _logger.info(f"Reading raw datasets from {cfg.dir.data_raw} ...")
    df_historical_weather = pl.read_csv(
        path_data_raw / "historical_weather.csv",
        try_parse_dates=True,
    )
    with open(path_data_raw / "county_id_to_name_map.json", "r") as file:
        county_id_to_name = json.load(file)
        county_name_to_id = {
            n.lower().capitalize(): int(i) for i, n in county_id_to_name.items()
        }
    df_capitals = (
        pl.read_csv(Path(cfg.dir.data_processed) / "download_geo_data" / "capitals.csv")
        .with_columns(pl.col("county_name").map_dict(county_name_to_id).alias("county"))
        .drop("county_name")
    )

    _logger.info(f"Finding relevant stations (within {cfg.max_distance}km, on_land_only={cfg.on_land_only}) ...")
    df_relevant_stations = find_relevant_county_stations(
        df_capitals=df_capitals,
        df_weather_hist=df_historical_weather,
        max_dist=cfg.max_distance,
        on_land_only=cfg.on_land_only,
    )

    _logger.info(f"Calculating stations weights (min_weight={cfg.min_weight}) ...")
    df_stations_with_weights = df_relevant_stations.group_by("county").map_groups(
        partial(calculate_weights_of_stations, min_weight=cfg.min_weight)
    ).drop_nulls("county")

    _logger.debug(df_stations_with_weights)

    path_df = path_data_processed / "stations_with_weights.parquet"
    _logger.info(f"Writing to {path_df}...")
    df_stations_with_weights.write_parquet(path_df)

if __name__ == "__main__":
    prepare_stations()