import abc
from typing_extensions import Self
import numpy as np
import pandas as pd


class ConsumptionRegressionBase(abc.ABC):
    @abc.abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pass

    @abc.abstractclassmethod
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> Self:
        pass
