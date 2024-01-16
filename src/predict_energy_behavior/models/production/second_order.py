from typing import Any
from mercantile import feature
import numpy as np
import pandas as pd
from predict_energy_behavior.models.production.base_model import (
    ProductionRegressionBase,
)
from sklearn.ensemble import VotingRegressor
import lightgbm as lgb


class SecondOrderModel(ProductionRegressionBase):
    def __init__(self, features: list[str], model: VotingRegressor) -> None:
        self.features = features
        self.model = model

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X[self.features])

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "SecondOrderModel":
        X_train = X[self.features]
        self.model.fit(X_train, y)
        return self


class LGBMSecondOrderModel(SecondOrderModel):
    def __init__(
        self, 
        features: list[str], 
        n_models: int, 
        parameters: dict[str, Any],
        n_jobs: int = 2
    ) -> None:
        model = VotingRegressor(
            estimators=[
                (f"regressor_{i}", lgb.sklearn.LGBMRegressor(**parameters, random_state=i))
                for i in range(n_models)
            ],
            n_jobs=n_jobs
        )
        super().__init__(features=features, model=model)
