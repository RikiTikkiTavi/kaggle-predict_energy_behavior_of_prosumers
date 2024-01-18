import logging
import pickle
import sklearn
from sklearn.ensemble import VotingRegressor
import predict_energy_behavior.config as config
import predict_energy_behavior.models.joined_model as joined_model
import predict_energy_behavior.utils as utils
import pandas as pd
from pathlib import Path
import hydra
import mlflow
import lightgbm

_logger = logging.getLogger(__name__)


def select_consumption(df: pd.DataFrame, is_consumption: bool = False) -> pd.DataFrame:
    return df.loc[df["is_consumption"] == int(is_consumption)]


@hydra.main(config_path="../../configs", config_name="train", version_base="1.3")
def main(cfg: config.ConfigExperiment):
    lightgbm.register_logger(_logger)

    if cfg.mlflow_tracking_uri is not None:
        mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)

    experiment = mlflow.set_experiment(cfg.exp_name)
    with mlflow.start_run(
        run_name=cfg.run_name, experiment_id=experiment.experiment_id
    ):
        mlflow.log_params(utils.flatten_dict(cfg))

        path_data_processed = Path(cfg.dir.data_processed)
        path_df = (
            path_data_processed / cfg.phase / "make_features" / "df_features.parquet"
        )

        _logger.info(f"Reading data from {path_df} ...")
        df_features = pd.read_parquet(path_df).dropna()
        val_start_date = pd.Timestamp.fromisoformat(cfg.split.val_start_date)

        model: joined_model.JoinedModel = hydra.utils.instantiate(cfg.model)

        df_train = df_features.loc[df_features["datetime"] < val_start_date]
        df_val = df_features.loc[df_features["datetime"] >= val_start_date]

        model.fit(
            train_tups=(
                (df_features, df_features["target"]),
                (df_train, df_train["target"]),
                (df_train, df_train["target"]),
            )
        )

        _logger.info("Validation ...")

        metrics = model.evaluate(
            df_val,
            df_val["target"],
            metrics={"val_MAE": sklearn.metrics.mean_absolute_error},
        )

        for submodel_key, submodel_metrics in metrics.items():
            for metric_name, metric_value in submodel_metrics.items():
                mlflow.log_metric(key=f"{submodel_key}/{metric_name}", value=metric_value, step=0)

        _logger.info(metrics)

        path_model = Path.cwd() / "model_joined.pickle"
        _logger.info(f"Saving consumption model to {path_model} ...")
        with open(path_model, "wb") as file:
            pickle.dump(model, file)

        return metrics["total"]["val_MAE"]


if __name__ == "__main__":
    main()
