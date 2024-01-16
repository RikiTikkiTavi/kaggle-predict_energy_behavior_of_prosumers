from typing import overload
import numpy as np
import pandas as pd
from predict_energy_behavior.models.production.base_model import ProductionRegressionBase
from predict_energy_behavior.models.production.solar_output_regressor import SolarOutputRegressor

import logging

_logger = logging.getLogger(__name__)

class MultiOrderRegression(ProductionRegressionBase):

    def __init__(
            self,
            first_order_model: ProductionRegressionBase,
            second_order_model: ProductionRegressionBase 
    ) -> None:
        self._first_order_model = first_order_model
        self._second_order_model = second_order_model

    @property
    def first_order(self) -> SolarOutputRegressor:
        return self._first_order_model
    
    @property
    def second_order(self) -> SolarOutputRegressor:
        return self._second_order_model

    def predict(self, X: pd.DataFrame):
        X = X.copy()
        
        results_first_order = self._first_order_model.predict(X)
        X["predictions_first_order"] = results_first_order

        return results_first_order + self._second_order_model.predict(X)

    def _fit_with_separate_dfs(self, d_1: tuple[pd.DataFrame, np.ndarray], d_2: tuple[pd.DataFrame, np.ndarray]):
        _logger.info("Fit 1 order ...")
        self._first_order_model.fit(*d_1)
        _logger.info(f"Fit 1 order model:\n{str(self._first_order_model)}")
        
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
    def fit(self, d_1: tuple[pd.DataFrame, np.ndarray], d_2: tuple[pd.DataFrame, np.ndarray]) -> "MultiOrderRegression": ...

    @overload
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "MultiOrderRegression": ...

    def fit(self, **kwargs) -> "MultiOrderRegression":
        if "d_1" in kwargs and "d_2" in kwargs:
            self._fit_with_separate_dfs(**kwargs)
        else:
            self._fit_with_same_df(**kwargs)

        return self

        