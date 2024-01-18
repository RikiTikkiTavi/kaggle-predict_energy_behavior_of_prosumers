import abc
from typing import Generic, Literal, TypeVar, overload
from typing_extensions import Self
import numpy as np
import pandas as pd


OrderType = TypeVar("OrderType", Literal[1], Literal[2])
TrainTupType = tuple[pd.DataFrame, np.ndarray]


class RegressionBase(abc.ABC, Generic[OrderType]):
    
    @abc.abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pass
    
    @overload
    @abc.abstractclassmethod
    def fit(
        self: "RegressionBase[Literal[2]]",
        train_tups: tuple[TrainTupType, TrainTupType, TrainTupType],
    ) -> "RegressionBase[Literal[2]]":
        ...

    @abc.abstractclassmethod
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> Self:
        ...