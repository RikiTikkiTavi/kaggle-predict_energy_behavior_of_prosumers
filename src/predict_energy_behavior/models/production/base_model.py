import abc
import numpy as np

import pandas as pd


class ProductionRegressionBase(abc.ABC):
    @abc.abstractmethod
    def predict(
        self, X: pd.DataFrame
    ) -> np.ndarray:
        pass

    @abc.abstractclassmethod
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "ProductionRegressionBase":
        pass