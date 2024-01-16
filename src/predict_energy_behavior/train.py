import logging
import pickle
import sklearn
from sklearn.ensemble import VotingRegressor
import predict_energy_behavior.config as config
import predict_energy_behavior.models.production.solar_output_regressor as solar_output_regressor
import predict_energy_behavior.models.production.multi_order_regression as multi_order_regression
import predict_energy_behavior.models.production.second_order as second_order
import pandas as pd
from pathlib import Path
import lightgbm as lgb
import hydra

_logger = logging.getLogger(__name__)


@hydra.main(config_path="../../configs", config_name="train", version_base="1.3")
def main(cfg: config.ConfigExperiment):
    path_data_processed = Path(cfg.dir.data_processed)
    path_df = path_data_processed / cfg.phase / "make_features" / "df_features.parquet"

    _logger.info(f"Reading data from {path_df} ...")
    df_features = pd.read_parquet(path_df).dropna()
    val_start_date = pd.Timestamp.fromisoformat(cfg.split.val_start_date)

    features_second_order = [
        "temperature",
        "dewpoint",
        "rain",
        "snowfall",
        "surface_pressure",
        "cloudcover_total",
        "cloudcover_low",
        "cloudcover_mid",
        "cloudcover_high",
        "windspeed_10m",
        "winddirection_10m",
        "shortwave_radiation",
        "direct_solar_radiation",
        "diffuse_radiation",
        "sin(hour)",
        "cos(hour)",
        "sin(dayofyear)",
        "cos(dayofyear)",
        "county",
        "is_business",
        "product_type",
        "installed_capacity",
        "predictions_first_order",
    ]

    second_order_params = {
        "objective": "regression_l1",
        "n_estimators": 2000,
        "learning_rate": 0.008,
        "colsample_bytree": 0.8,
        "colsample_bynode": 0.5,
        "lambda_l1": 3.4,
        "lambda_l2": 1.4,
        "max_depth": 24,
        "num_leaves": 490,
        "min_data_in_leaf": 48,
        "device": "cuda"
    }

    model_p = multi_order_regression.MultiOrderRegression(
        first_order_model=solar_output_regressor.SolarOutputRegressor(),
        second_order_model=second_order.LGBMSecondOrderModel(
            features=features_second_order, n_models=3, parameters=second_order_params, n_jobs=1,
        ),
    )

    df_train = df_features.loc[df_features["datetime"] < val_start_date]
    df_val = df_features.loc[df_features["datetime"] >= val_start_date]

    df_train_p = df_train[df_train["is_consumption"] == 0]
    df_val_p = df_val[df_val["is_consumption"] == 0]

    _logger.info(f"Fitting model ...")
    model_p.fit(df_train_p, df_train_p["target"])

    _logger.info("Validation ...")
    _logger.info(
        f"Val MAE (1): {sklearn.metrics.mean_absolute_error(model_p.first_order.predict(df_val_p), df_val_p['target'])}"
    )
    preds_val = model_p.predict(df_val_p)
    val_score = sklearn.metrics.mean_absolute_error(preds_val, df_val_p["target"])
    _logger.info(f"Val MAE (total): {val_score}")

    path_model_p = Path.cwd() / "model_p.pickle"
    _logger.info(f"Saving production model to {path_model_p} ...")
    with open(path_model_p):
        pickle.dump(model_p)

    return val_score


if __name__ == "__main__":
    main()
