import numpy as np
import pandas as pd
from predict_energy_behavior.models.production.base_model import ProductionRegressionBase
from predict_energy_behavior.models.production.solar_output_regressor import SolarOutputRegressor

import logging

_logger = logging.getLogger(__name__)

class MultiOrderRegression(ProductionRegressionBase):

    def __init__(
            self,
            first_order_model: SolarOutputRegressor,
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

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "MultiOrderRegression":
        X = X.copy()

        _logger.info("Fit 1 order ...")
        self._first_order_model.fit(X, y)
        _logger.info(f'1 order model train metric: {self._first_order_model.loss.__name__}={self._first_order_model.optim_result.fun}')
        _logger.info(f'1 order model weights: {self._first_order_model.weights}')
        
        results_first_order = self._first_order_model.predict(X)
        X["predictions_first_order"] = results_first_order
        target_second_order = y - results_first_order
        
        _logger.info("Fit 2 order ...")
        self._second_order_model.fit(X, target_second_order)

        return self

        