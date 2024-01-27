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

def is_prediciton_needed(df_test):
    return not all(df_test['currently_scored'] == False)

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
        t0 = time.time()
        
        try:
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
        except Exception as e:
            df_sample_prediction["target"] = 10_000_000
            print(e)
            env.predict(df_sample_prediction)
            continue

        if not cfg.debug:
            if not is_prediciton_needed(df_test):
                df_sample_prediction["target"] = 0.0
                env.predict(df_sample_prediction)
                continue

        t_read = time.time()
        _logger.info(f"Time to read: {t_read-t0}s")

        try:
            # generate test features
            df_test = ds.preprocess_test(df_test)
            df_test_features = feature_gen.generate_features(df_test)
            df_test_features = train.replace_historical_with_forecast(df_test_features)
        except Exception as e:
            df_sample_prediction["target"] = 30_000_000
            print(e)
            env.predict(df_sample_prediction)
            continue
        
        t_process = time.time()
        _logger.info(f"Time to process: {t_process-t_read}s")

        try:
            preds = model.predict(df_test_features).clip(0)
            df_sample_prediction["target"] = preds
        except Exception as e:
            df_sample_prediction["target"] = 50_000_000
            print(e)
            env.predict(df_sample_prediction)
            continue

        try:
            assert not df_sample_prediction["target"].isna().any()
        except Exception as e:
            df_sample_prediction["target"] = 70_000_000
            print(e)
            env.predict(df_sample_prediction)
            continue

        try:
            assert not np.isinf(preds).any()
        except Exception as e:
            df_sample_prediction["target"] = 90_000_000
            print(e)
            env.predict(df_sample_prediction)
            continue

        if cfg.debug:
            assert not df_sample_prediction["target"].isna().any()
            assert not np.isinf(preds).any()

        t_predict = time.time()
        _logger.info(f"Time to predict: {t_predict-t_process}s")

        env.predict(df_sample_prediction)


if __name__ == "__main__":
    main()
