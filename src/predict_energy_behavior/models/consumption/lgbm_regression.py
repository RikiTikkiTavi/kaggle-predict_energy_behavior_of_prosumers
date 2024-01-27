from pathlib import Path
from typing import Any
import joblib
from sklearn.ensemble import VotingRegressor
from predict_energy_behavior.models.consumption.base_model import (
    ConsumptionRegressionBase,
)
import lightgbm as lgb

import numpy as np
import pandas as pd

class LGBMSecondOrderModel(ConsumptionRegressionBase):
    def __init__(
        self,
        features: list[str],
        n_models: int,
        parameters: dict[str, Any],
        n_jobs: int = 2,
        n_gpus: int = 4,
    ) -> None:
        model = VotingRegressor(
            estimators=[
                (
                    f"regressor_{i}",
                    lgb.sklearn.LGBMRegressor(
                        **parameters, random_state=i, gpu_device_id=i % n_gpus
                    ),
                )
                for i in range(n_models)
            ],
            n_jobs=n_jobs,
        )
        super().__init__(features=features, model=model)

    @classmethod
    def load(cls, path: Path, name_prefix="consumption_2_") -> "LGBMSecondOrderModel":
        m = joblib.load(path / f"{name_prefix}model_obj.pickle")
        for estimator_name, estimator in m.model.named_estimators_.items():
            model_path = path / f"{name_prefix}{estimator_name}.txt"
            new_booster = lgb.Booster(model_file=model_path)
            if hasattr(estimator, "_Booster"):
                estimator._Booster = new_booster
        return m

    def save(self, path: Path, name_prefix="consumption_2_"):
        joblib.dump(self, path / f"{name_prefix}model_obj.pickle")
        for name, estimator in self.model.named_estimators_.items():
            # Check if the estimator is LGBMRegressor and has been fitted
            if hasattr(estimator, "booster_"):
                model_path = path / f"{name_prefix}{name}.txt"
                estimator.booster_.save_model(model_path)


class LGBMOnDiff(LGBMSecondOrderModel):
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X = X.assign(
            **{
                "predictions_production": np.where(
                    X["installed_capacity"] > 0,
                    X["predictions_production"] / X["installed_capacity"],
                    X["predictions_production"],
                )
            }
        )

        return self.model.predict(X[self.features]) * X["eic_count"]

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "ConsumptionRegressionBase":
        assert (
            "predictions_production" in X.columns and "installed_capacity" in X.columns
        )
        assert "eic_count" in X.columns

        X = X.assign(
            **{
                "predictions_production": np.where(
                    X["installed_capacity"] > 0,
                    X["predictions_production"] / X["installed_capacity"],
                    X["predictions_production"],
                )
            }
        )

        y = np.where(X["eic_count"] > 0, y / X["eic_count"], y)

        y -= X["target_per_eic_48h"]

        X_train = X[self.features]
        self.model.fit(X_train, y)

        return self
