import logging
import pickle
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

_logger = logging.getLogger(__name__)


def select_consumption(df: pd.DataFrame, is_consumption: bool = False) -> pd.DataFrame:
    return df.loc[df["is_consumption"] == int(is_consumption)]

def replace_historical_with_forecast(df: pd.DataFrame) -> pd.DataFrame:
    features = df.columns
    historical_weather_features = [f for f in features if f.endswith("historical")]
    corresponding_weather_features  = [f.replace("_historical", "_forecast") for f in historical_weather_features]
    df = df.drop(columns=historical_weather_features)
    df = df.rename({f_c: f_h for f_h, f_c in zip(historical_weather_features, corresponding_weather_features)}, axis=1)
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

        val_start_date = pd.Timestamp.fromisoformat(cfg.split.val_start_date)

        model: joined_model.JoinedModel = hydra.utils.instantiate(cfg.model)

        df_train = df_features.loc[df_features["datetime"] < val_start_date]
        df_val = df_features.loc[df_features["datetime"] >= val_start_date]

        print(df_val.dtypes.to_dict())

        model.fit(
            train_tups=(
                (df_features, df_features["target"]),
                (df_train, df_train["target"]),
                (df_train, df_train["target"]),
            )
        )

        path_model = Path.cwd() / "model_joined"
        model.save(path_model)

        _logger.info("Validation on historical weather ...")

        metrics_hist = model.evaluate(
            df_val,
            df_val["target"],
            metrics={"hist_val_MAE": sklearn.metrics.mean_absolute_error},
        )

        for submodel_key, submodel_metrics in metrics_hist.items():
            for metric_name, metric_value in submodel_metrics.items():
                mlflow.log_metric(key=f"{submodel_key}/{metric_name}", value=metric_value, step=0)

        _logger.info(metrics_hist)

        _logger.info("Reloading model ...")
        model = joined_model.JoinedModel.load(path_model)
        _logger.info(
            model.evaluate(
                df_val,
                df_val["target"],
                metrics={"hist_val_MAE": sklearn.metrics.mean_absolute_error},
            )
        )
        

        _logger.info("Validation on forecast ...")

        _logger.info("Replacing historical features with forecast features in val df ...")
        df_val = replace_historical_with_forecast(df_val)

        _logger.info(list(df_val.columns))

        metrics_forecast = model.evaluate(
            df_val,
            df_val["target"],
            metrics={"forecast_val_MAE": sklearn.metrics.mean_absolute_error},
        )

        _logger.info(metrics_forecast)

        return metrics_hist["total"]["hist_val_MAE"]


if __name__ == "__main__":
    main()
