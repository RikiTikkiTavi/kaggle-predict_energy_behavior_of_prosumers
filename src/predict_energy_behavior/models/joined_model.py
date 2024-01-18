import logging
from typing import Any, Callable, Literal, Sequence, overload
from typing_extensions import Self
import numpy as np
import pandas as pd
from predict_energy_behavior.models.base_model import RegressionBase
from predict_energy_behavior.models.production.base_model import (
    ProductionRegressionBase,
    TrainTupType,
)
from predict_energy_behavior.models.consumption.base_model import (
    ConsumptionRegressionBase,
)

_logger = logging.getLogger(__name__)


def select_consumption(df: pd.DataFrame, is_consumption: bool = False) -> pd.DataFrame:
    return df.loc[df["is_consumption"] == int(is_consumption)]


class JoinedModel(RegressionBase[Literal[2]]):
    def __init__(
        self,
        model_p: ProductionRegressionBase[Literal[2]],
        model_c: ConsumptionRegressionBase,
    ) -> None:
        super().__init__()
        self._model_p = model_p
        self._model_c = model_c

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_p = select_consumption(X, False)
        preds_p = pd.Series(self._model_p.predict(X_p), index=X_p.index)

        X_c = select_consumption(X, True)
        preds_c = pd.Series(self._model_c.predict(X_c), index=X_c.index)

        return pd.concat(preds_p, preds_c).loc[X.index].to_numpy()

    @overload
    def fit(
        self,
        train_tups: tuple[TrainTupType, TrainTupType, TrainTupType],
    ) -> Self:
        ...

    def _fit_different_dfs(
        self, train_tups: tuple[TrainTupType, TrainTupType, TrainTupType]
    ) -> Self:
        y_1 = pd.Series(train_tups[0][1], index=train_tups[0][0].index)
        y_2 = pd.Series(train_tups[1][1], index=train_tups[1][0].index)
        y_3 = pd.Series(train_tups[2][1], index=train_tups[2][0].index)

        _logger.info(f"Fit production model ...")

        X_p_1 = select_consumption(train_tups[0][0], False)
        y_p_1 = y_1.loc[X_p_1.index].to_numpy()
        X_p_2 = select_consumption(train_tups[1][0], False)
        y_p_2 = y_2.loc[X_p_2.index].to_numpy()

        self._model_p.fit(train_tups=((X_p_1, y_p_1), (X_p_2, y_p_2)))

        _logger.info("Fit consumption model ...")

        X_c = select_consumption(train_tups[2][0], True)
        y_c = y_3.loc[X_c.index].to_numpy()
        self._model_c.fit(X_c, y_c)

        return self

    @overload
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> Self:
        ...

    def _fit_same_df(self, X: pd.DataFrame, y: np.ndarray) -> Self:
        y = pd.Series(y, index=X.index)

        X_p = select_consumption(X, False)
        y_p = y.loc[X_p.index].to_numpy()
        self._model_p.fit(X_p, y_p)

        X_c = select_consumption(X, True)
        y_c = y.loc[X_c.index].to_numpy()
        self._model_c.fit(X_c, y_c)

        return self

    def fit(self, **kwargs) -> Self:
        if "X" in kwargs and "y" in kwargs:
            self._fit_same_df(**kwargs)
        elif "train_tups" in kwargs:
            self._fit_different_dfs(**kwargs)
        return self

    def evaluate(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        metrics: dict[str, Callable[[np.ndarray, np.ndarray], float]],
    ) -> dict[str, dict[str, float]]:
        y = pd.Series(y, index=X.index)

        X_p = select_consumption(X, False)
        preds_p = pd.Series(self._model_p.predict(X_p), index=X_p.index)

        X_c = select_consumption(X, True)
        preds_c = pd.Series(self._model_c.predict(X_c), index=X_c.index)

        preds = pd.concat([preds_p, preds_c]).loc[X.index].to_numpy()

        return {
            "production": {key: metric(preds_p, y.loc[X_p.index]) for key, metric in metrics.items()},
            "consumption": {key: metric(preds_c, y.loc[X_c.index]) for key, metric in metrics.items()},
            "total": {key: metric(preds, y) for key, metric in metrics.items()}
        }