import sklearn
from sklearn.ensemble import VotingRegressor
import predict_energy_behavior.config as config
import predict_energy_behavior.models.solar_output_regressor as solar_output_regressor
import pandas as pd
from pathlib import Path
import lightgbm as lgb
import hydra

@hydra.main(config_path="../../configs", config_name="train", version_base="1.2")
def main(cfg: config.ConfigExperiment):
    path_data_processed = Path(cfg.dir.data_processed)

    df_features = pd.read_parquet(path_data_processed / cfg.phase / "make_features" / "df_train_features.parquet").dropna()

    _parameters = [
        solar_output_regressor.Parameter(name="C_area", value=2.0, bounds=(0.0, 10.0)),
        solar_output_regressor.Parameter(name="C_fog", value=0.3, bounds=(0.0, 1.0)),
        solar_output_regressor.Parameter(name="C_dew", value=0.95, bounds=(0.0, 1.0)),
        solar_output_regressor.Parameter(name="C_rain", value=0.5, bounds=(0.1, 1.0)),
        solar_output_regressor.Parameter(name="C_rad_diffuse", value=0.05, bounds=(0.0, 1.0)),
        solar_output_regressor.Parameter(name="C_rad_direct_lin", value=669.0, bounds=(650.0, 700)),
        solar_output_regressor.Parameter(name="C_rad_direct_scale", value=1.21, bounds=(1, 1.3)),
        solar_output_regressor.Parameter(name=f"C_snow", value=10.0, bounds=(0.0, 100.0))
    ]

    val_start_date = pd.Timestamp(year=2023, month=1, day=1)

    df_features_train_p = df_features.loc[(df_features["datetime"]<val_start_date) & (df_features["is_consumption"] == 0)]
    df_features_val_p = df_features.loc[(df_features["datetime"]>val_start_date) & (df_features["is_consumption"] == 0)]
    
    p_regressor_1 = solar_output_regressor.SolarOutputRegressor.from_params(
        regressors=[],
        params=_parameters,
        method="Nelder-Mead"
    )

    p_regressor_1.fit(df_features_train_p, df_features_train_p["target"])
    print(f"Score: {p_regressor_1.loss.__name__}={p_regressor_1.optim_result.fun}")

    X_train = df_features_train_p.drop(columns=["target", "datetime", "date", "segment"])
    X_val = df_features_val_p.drop(columns=["target", "datetime", "date", "segment"])

    X_train["model_predictions"] = p_regressor_1.predict(X_train)
    X_val["model_predictions"] = p_regressor_1.predict(X_val)

    model_parameters_p = {
            "n_estimators": 2500,
            "learning_rate": 0.052587652,
            "colsample_bytree": 0.9,
            "colsample_bynode": 0.6,
            "lambda_l1": 3.5,
            "lambda_l2": 1.5,
            "max_depth": 16,
            "num_leaves": 500,
            "min_data_in_leaf": 50,
            "objective": "regression_l1",
            "device_type": "cuda",
        }

    model_production = VotingRegressor([
        (f'lgb_{i}', lgb.LGBMRegressor(**model_parameters_p, random_state=i)) for i in range(10)
    ])

    preds = model_production.fit(X_train, df_features_train_p["target"])

    print("MAE: ", sklearn.metrics.mean_absolute_error(preds, df_features_val_p["target"]))

if __name__ == "__main__":
    main()