import abc
from typing_extensions import Self
import numpy as np
import pandas as pd
from sklearn.ensemble import VotingRegressor


class ConsumptionRegressionBase(abc.ABC):
    def __init__(self, features: list[str], model: VotingRegressor) -> None:
        self.features = features
        self.model = model

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

        y  = np.where(
            X["eic_count"]>0,
            y / X["eic_count"],
            y
        )

        X_train = X[self.features]
        self.model.fit(X_train, y)

        return self
