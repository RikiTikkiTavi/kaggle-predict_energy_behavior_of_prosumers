import logging
import pickle
import numpy as np
import sklearn
from sklearn.ensemble import VotingRegressor
import predict_energy_behavior.config as config
import predict_energy_behavior.models.joined_model as joined_model
import predict_energy_behavior.utils.common as common
import pandas as pd
from pathlib import Path
import hydra
import mlflow
import lightgbm

from typing import Generator

_logger = logging.getLogger(__name__)


def select_consumption(df: pd.DataFrame, is_consumption: bool = False) -> pd.DataFrame:
    return df.loc[df["is_consumption"] == int(is_consumption)]


def replace_historical_with_forecast(df: pd.DataFrame) -> pd.DataFrame:
    features = df.columns
    historical_weather_features = [f for f in features if f.endswith("historical")]
    corresponding_weather_features = [
        f.replace("_historical", "_forecast") for f in historical_weather_features
    ]
    df = df.drop(columns=historical_weather_features)
    df = df.rename(
        {
            f_c: f_h
            for f_h, f_c in zip(
                historical_weather_features, corresponding_weather_features
            )
        },
        axis=1,
    )
    return df


@hydra.main(config_path="../../configs", config_name="train", version_base="1.3")
def main(cfg: config.ConfigExperiment):
    lightgbm.register_logger(_logger)

    if cfg.mlflow_tracking_uri is not None:
        mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)

    experiment = mlflow.set_experiment(cfg.exp_name)
    with mlflow.start_run(
        run_name=cfg.run_name, experiment_id=experiment.experiment_id
    ):
        mlflow.log_params(common.flatten_dict(cfg))

        path_data_processed = Path(cfg.dir.data_processed)
        path_df = (
            path_data_processed / cfg.phase / "make_features" / "df_features.parquet"
        )

        _logger.info(f"Reading data from {path_df} ...")
        df_features = pd.read_parquet(path_df, engine="fastparquet").dropna()

        _logger.info(list(df_features.columns))

        cv = hydra.utils.instantiate(cfg.cv)

        metrics = {}
        models = {}

        for fold_i, df_train, df_val in cv.split(df_features):
            _logger.info(f"\n-------------- Fold {fold_i} --------------\n")
            
            model: joined_model.JoinedModel = hydra.utils.instantiate(cfg.model)

            model.fit(
                train_tups=(
                    (df_train, df_train["target"]),
                    (df_train, df_train["target"]),
                    (df_train, df_train["target"]),
                )
            )

            models[fold_i] = model

            path_model = Path.cwd() / "model_joined"
            model.save(path_model)

            _logger.info("Validation on historical weather ...")

            fold_metrics_hist = model.evaluate(
                df_val,
                df_val["target"],
                metrics={"MAE": sklearn.metrics.mean_absolute_error},
            )

            for submodel_key, submodel_metrics in fold_metrics_hist.items():
                for metric_name, metric_value in submodel_metrics.items():
                    mlflow.log_metric(
                        key=f"{fold_i}/h/{submodel_key}/{metric_name}", value=metric_value, step=0
                    )

            _logger.info(f"Historical metrics on fold={fold_i}: \n{fold_metrics_hist}")

            _logger.info("Validation on forecast ...")

            _logger.info(
                "Replacing historical features with forecast features in val df ..."
            )
            df_val = replace_historical_with_forecast(df_val)

            fold_metrics_forecast = model.evaluate(
                df_val,
                df_val["target"],
                metrics={"MAE": sklearn.metrics.mean_absolute_error},
            )

            for submodel_key, submodel_metrics in fold_metrics_forecast.items():
                for metric_name, metric_value in submodel_metrics.items():
                    mlflow.log_metric(
                        key=f"{fold_i}/f/{submodel_key}/{metric_name}", value=metric_value, step=0
                    )

            _logger.info(f"Forecast metrics on fold={fold_i}: \n{fold_metrics_forecast}")

            metrics[fold_i] = {
                "h": fold_metrics_hist,
                "f": fold_metrics_forecast
            }

        total_avg_forecast = np.mean([
            metrics_fold["f"]["total"]["MAE"] for metrics_fold in metrics.values()
        ])

        mlflow.log_metric(key="total/f/MAE", value=total_avg_forecast, step=0)
        _logger.info(f"total/f/MAE={total_avg_forecast}")

        return total_avg_forecast


if __name__ == "__main__":
    main()
