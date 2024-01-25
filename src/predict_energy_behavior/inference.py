import logging
import pickle
import time
import numpy as np
import sklearn
from sklearn.ensemble import VotingRegressor
import predict_energy_behavior.config as config
import predict_energy_behavior.models.joined_model as joined_model
import predict_energy_behavior.data.data_storage as data_storage
import predict_energy_behavior.features.feature_generator as feature_generator
import predict_energy_behavior.train as train
import predict_energy_behavior.utils.common as common
import pandas as pd
from pathlib import Path
import hydra
import predict_energy_behavior.enefit.public_timeseries_testing_util
import gc
import sys

_logger = logging.getLogger(__name__)


def date_columns_to_datetime(df):
    columns = ["forecast_date", "forecast_datetime", "datetime", "prediction_datetime", "origin_datetime"]
    for col in columns:
        if col in df.columns and df[col].dtype == "object":
            df[col] = pd.to_datetime(df[col])
    return df


@hydra.main(config_path="../../configs", config_name="inference", version_base="1.3")
def main(cfg: config.ConfigInference):
    _logger.info(f"Loading model {cfg.model} ...")
    model = hydra.utils.instantiate(cfg.model)
    
    ds = data_storage.DataStorage(
        path_data_raw=Path(cfg.dir.data_raw),
        path_data_geo=Path(cfg.dir.data_geo),
        path_data_stations=Path(cfg.dir.data_stations),
    )
    feature_gen = feature_generator.FeaturesGenerator(ds, {})

    if cfg.debug:
        env = predict_energy_behavior.enefit.public_timeseries_testing_util.MockApi(Path(cfg.dir.data_raw) / "example_test_files")
        iter_test = env.iter_test()
    else:
        sys.path.append(f"{cfg.dir.data_raw}/enefit")
        import enefit
        env = enefit.make_env()
        iter_test = env.iter_test()

    for (
        df_test,
        df_new_target,
        df_new_client,
        df_new_historical_weather,
        df_new_forecast_weather,
        df_new_electricity_prices,
        df_new_gas_prices,
        df_sample_prediction,
    ) in iter_test:
        if not cfg.debug:
            if not bool(df_test["currently_scored"].iloc[0]):
                df_sample_prediction["target"] = 0.0
                env.predict(df_sample_prediction)
                continue

        df_new_target["target"] = df_new_target["target"].fillna(0.0)
        df_new_client["installed_capacity"] = df_new_client["installed_capacity"].fillna(1000.0)
        df_new_client["eic_count"] = df_new_client["eic_count"].fillna(100)

        t0 = time.time()

        date_columns_to_datetime(df_new_electricity_prices)
        date_columns_to_datetime(df_new_forecast_weather)
        date_columns_to_datetime(df_new_historical_weather)
        date_columns_to_datetime(df_new_target)
        date_columns_to_datetime(df_test)

        ds.update_with_new_data(
            df_new_client=df_new_client,
            df_new_gas_prices=df_new_gas_prices,
            df_new_electricity_prices=df_new_electricity_prices,
            df_new_forecast_weather=df_new_forecast_weather,
            df_new_historical_weather=df_new_historical_weather,
            df_new_target=df_new_target,
        )

        t_read = time.time()
        _logger.info(f"Time to read: {t_read-t0}s")

        # separately generate test features for both models

        df_test = ds.preprocess_test(df_test)
        df_test_features = feature_gen.generate_features(df_test)

        df_test_features = train.replace_historical_with_forecast(df_test_features)

        t_process = time.time()
        _logger.info(f"Time to process: {t_process-t_read}s")

        preds = model.predict(df_test_features).clip(0)
        df_sample_prediction["target"] = preds

        if cfg.debug:
            assert not np.isnan(preds).any()

        t_predict = time.time()
        _logger.info(f"Time to predict: {t_predict-t_process}s")

        env.predict(df_sample_prediction)
        gc.collect()


if __name__ == "__main__":
    main()
