import abc
from typing import Generic, Literal, NewType, Optional, TypeVar, overload
from typing_extensions import Self
import numpy as np
import pandas as pd

OrderType = TypeVar("OrderType", Literal[1], Literal[2])
TrainTupType = tuple[pd.DataFrame, np.ndarray]


class ProductionRegressionBase(Generic[OrderType]):
    @abc.abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pass
    
    @abc.abstractclassmethod
    def set_regressors(
        self: "ProductionRegressionBase[Literal[1]]", 
        regressors: dict[str, Optional[str]]
    ):
        ...

    @abc.abstractclassmethod
    def get_model(
        self, 
        order: int
    ) -> "ProductionRegressionBase[Literal[1]]":
        ...

    @overload
    def fit(
        self: "ProductionRegressionBase[Literal[2]]",
        train_tups: tuple[TrainTupType, TrainTupType],
    ) -> "ProductionRegressionBase[Literal[2]]":
        ...


    @abc.abstractclassmethod
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> Self:
        ...
