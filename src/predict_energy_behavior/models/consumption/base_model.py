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
        
        X = X.assign(**{
            "predictions_production": X["predictions_production"] / X["installed_capacity"]
        })

        return self.model.predict(X[self.features])

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "ConsumptionRegressionBase":
        assert "predictions_production" in X.columns and "installed_capacity" in X.columns
        
        X = X.assign(**{
            "predictions_production": X["predictions_production"] / X["installed_capacity"]
        })

        X_train = X[self.features]
        self.model.fit(X_train, y)
        
        return self
