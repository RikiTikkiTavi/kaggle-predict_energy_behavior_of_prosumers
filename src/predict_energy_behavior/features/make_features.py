from pathlib import Path
import select
from typing import Sequence
import polars as pl
import numpy as np
import predict_energy_behavior.data.read as read
import predict_energy_behavior.data.constants as data_const
import predict_energy_behavior.config as config
import predict_energy_behavior.features.process as process
from predict_energy_behavior.features.feature_generator import FeaturesGenerator
from predict_energy_behavior.data.data_storage import DataStorage

from functools import partial

import hydra
import logging

import pandas as pd

_logger = logging.getLogger(__file__)


@hydra.main(
    config_path="../../../configs", config_name="prepare_data", version_base="1.3"
)
def prepare_data(cfg: config.ConfigPrepareData):
    _logger.info("Preparing data ...")
    path_data_processed = Path.cwd()

    _logger.info(f"Reading raw datasets from {cfg.dir.data_raw} ...")
    ds = DataStorage(
        path_data_raw=Path(cfg.dir.data_raw),
        path_data_geo=Path(cfg.dir.data_processed) / "download_geo_data",
    )

    _logger.info("Processing data ...")
    df: pd.DataFrame = FeaturesGenerator(ds, cfg).generate_features(ds.df_data)

    _logger.info(f"Features: {list(df.columns)}")

    path_data_processed.mkdir(exist_ok=True, parents=True)
    path_df_train = path_data_processed / "df_features.parquet"
    _logger.info(f"Saving processed data to {path_df_train} ...")
    df.to_parquet(path_df_train)
    


if __name__ == "__main__":
    prepare_data()
