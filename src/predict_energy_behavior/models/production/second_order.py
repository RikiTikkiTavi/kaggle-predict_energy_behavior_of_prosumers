from pathlib import Path
from typing import Any, Literal, Optional
import joblib
from mercantile import feature
import numpy as np
from numpy import ndarray
import pandas as pd
from predict_energy_behavior.models.production.base_model import (
    ProductionRegressionBase,
)
from sklearn.ensemble import VotingRegressor
import lightgbm as lgb


class SecondOrderModel(ProductionRegressionBase[Literal[1]]):
    def __init__(self, features: list[str], model: VotingRegressor) -> None:
        self.features = features
        self.model = model

    def set_regressors(
        self: "ProductionRegressionBase[Literal[1]]",
        regressors: dict[str, Optional[str]],
    ):
        self.features = list(regressors.values())

    def get_model(self, order: int):
        assert order == 1
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        
        X = X.assign(**{
            "predictions_first_order": X["predictions_first_order"] / X["installed_capacity"]
        })

        return self.model.predict(X[self.features]) * X["installed_capacity"]

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "SecondOrderModel":
        assert "predictions_first_order" in X.columns and "installed_capacity" in X.columns
        
        X["predictions_first_order"] /= X["installed_capacity"]
        y /= X["installed_capacity"]
        
        X_train = X[self.features]
        self.model.fit(X_train, y)
        
        return self


class LGBMSecondOrderModel(SecondOrderModel):
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
    def load(cls, path: Path, name_prefix="production_2_") -> "LGBMSecondOrderModel":
        m = joblib.load(path / f"{name_prefix}model_obj.pickle")
        for estimator_name, estimator in m.model.named_estimators_.items():
            model_path = path / f"{name_prefix}{estimator_name}.txt"
            new_booster = lgb.Booster(model_file=model_path)
            if hasattr(estimator, "_Booster"):
                estimator._Booster = new_booster
        return m

    def save(self, path: Path, name_prefix="production_2_"):
        joblib.dump(self, path / f"{name_prefix}model_obj.pickle")
        for name, estimator in self.model.named_estimators_.items():
            # Check if the estimator is LGBMRegressor and has been fitted
            if hasattr(estimator, "booster_"):
                model_path = path / f"{name_prefix}{name}.txt"
                estimator.booster_.save_model(model_path)