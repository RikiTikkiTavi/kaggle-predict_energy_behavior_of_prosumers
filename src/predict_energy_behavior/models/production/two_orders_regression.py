from pathlib import Path
from typing import Literal, overload
from typing_extensions import Self
import joblib
import numpy as np
import pandas as pd
from predict_energy_behavior.models.production.base_model import (
    ProductionRegressionBase,
    TrainTupType,
)
from predict_energy_behavior.models.production.solar_output_regression import (
    SolarOutputRegression,
    GroupedSolarOutputRegression
)

from predict_energy_behavior.models.production.second_order import (
    SecondOrderModel,
    LGBMSecondOrderModel
)

import logging

_logger = logging.getLogger(__name__)


class TwoOrdersRegression(ProductionRegressionBase[Literal[2]]):
    def __init__(
        self,
        first_order_model: LGBMSecondOrderModel,
        second_order_model: SecondOrderModel,
    ) -> None:
        self._first_order_model = first_order_model
        self._second_order_model = second_order_model

    def get_model(self, order: int):
        assert 0 < order and order < 3
        if order == 1:
            return self._first_order_model
        else:
            return self._second_order_model

    def predict(self, X: pd.DataFrame):
        X = X.copy()

        results_first_order = self._first_order_model.predict(X)
        X["predictions_first_order"] = results_first_order

        return results_first_order + self._second_order_model.predict(X)

    def _fit_with_separate_dfs(self, train_tups: tuple[TrainTupType, TrainTupType]):
        d_1, d_2 = train_tups
        
        _logger.info("Fit 1 order ...")
        self._first_order_model.fit(*d_1)
        # _logger.info(f"Fit 1 order model:\n{str(self._first_order_model)}")

        X = d_2[0].copy()
        results_first_order = self._first_order_model.predict(X)
        X["predictions_first_order"] = results_first_order
        target_second_order = d_2[1] - results_first_order

        _logger.info("Fit 2 order ...")
        self._second_order_model.fit(X, target_second_order)

        return self

    def _fit_with_same_df(self, X: pd.DataFrame, y: np.ndarray):
        X = X.copy()

        _logger.info("Fit 1 order ...")
        self._first_order_model.fit(X, y)
        _logger.info(f"Fit 1 order model:\n{str(self._first_order_model)}")

        results_first_order = self._first_order_model.predict(X)

        X["predictions_first_order"] = results_first_order
        target_second_order = y - results_first_order

        _logger.info("Fit 2 order ...")
        self._second_order_model.fit(X, target_second_order)

        return self

    @overload
    def fit(self, train_tups: tuple[TrainTupType, TrainTupType]) -> Self:
        ...

    @overload
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> Self:
        ...

    def fit(self, **kwargs) -> Self:
        if "train_tups" in kwargs:
            return self._fit_with_separate_dfs(**kwargs)
        else:
            return self._fit_with_same_df(**kwargs)

    @classmethod
    def load(cls, path: Path) -> "TwoOrdersRegression":
        return TwoOrdersRegression(
            first_order_model=GroupedSolarOutputRegression.load(path / "production_1.pickle"),
            second_order_model=LGBMSecondOrderModel.load(path)
        )

    def save(self, path: Path):
        self._first_order_model.save(path / "production_1.pickle")
        self._second_order_model.save(path)
