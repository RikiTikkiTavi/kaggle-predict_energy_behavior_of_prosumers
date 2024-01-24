from pathlib import Path
from numpy import ndarray
from scipy.optimize import minimize, OptimizeResult
import sklearn.metrics
from typing import Any, Callable, Literal, NoReturn, Sequence, Optional, TypedDict
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from sklearn.metrics._regression import mean_absolute_error as mean_absolute_error
from predict_energy_behavior.models.production.base_model import (
    ProductionRegressionBase,
)

import joblib
import multiprocessing


def calculate_relative_humidity(T, D):
    """
    Calculate the relative humidity based on temperature and dewpoint.
    T: Temperature in Celsius
    D: Dewpoint in Celsius
    """
    return 100 * (
        np.exp((17.625 * D) / (243.04 + D)) / np.exp((17.625 * T) / (243.04 + T))
    )


def wind_factor(WS, threshold=2.0):
    """
    Calculate the wind factor for fog formation.
    WS: Wind Speed in m/s
    threshold: Wind speed threshold for fog formation
    """
    # Wind factor reduces as wind speed increases. If WS is below the threshold, set the factor to 1.
    return np.where(WS < threshold, 1.0, 1 / WS)


def estimate_fog_intensity(T, D, WS):
    """
    Estimate the fog intensity based on temperature, dewpoint, and wind speed.
    T: Temperature in Celsius
    D: Dewpoint in Celsius
    WS: Wind Speed in m/s
    """
    RH = calculate_relative_humidity(T, D)
    WF = wind_factor(WS)

    # Assuming fog intensity is higher with higher relative humidity and lower wind speeds
    # This is a simple model and might need calibration with empirical data
    fog_intensity = RH * WF / 100  # Normalize to a 0-100 scale

    return fog_intensity


@dataclass
class Parameter:
    name: str
    value: float
    bounds: tuple[float, float]
    fixed: bool = False


@dataclass
class SolarOutputRegressors:
    temperature: Optional[str] = "temperature"
    snowfall: Optional[str] = "snowfall"
    rain: Optional[str] = "rain"
    dewpoint: Optional[str] = "dewpoint"
    windspeed: Optional[str] = "windspeed_10m"
    shortwave_radiation: str = "shortwave_radiation"
    # direct_solar_radiation: str = "direct_solar_radiation"
    # diffuse_radiation: Optional[str] = "diffuse_radiation"
    installed_capacity: str = "installed_capacity"


default_regressors = asdict(SolarOutputRegressors())


class SolarOutputRegression(ProductionRegressionBase[Literal[1]]):
    regressors: dict[str, Optional[str]]
    weights: dict[str, float]
    bounds: dict[str, tuple[float, float]]
    baseline_temperature: float
    baseline_efficiency: float
    loss: Callable[[np.ndarray, np.ndarray], float]
    optim_result: Optional[OptimizeResult] = None

    weights = {
        "C_area": 4.046345258643658,
        "C_fog": 0.14347390597725973,
        "C_dew": 0.653589763694636,
        "C_rain": 0.1,
        "C_rad_tot_k1": 0.3,
        "C_rad_tot_k2": 0.74,
        "C_rad_tot_k3": 0.4,
        "C_rad_tot_b1": 0.097,
        "C_rad_tot_b2": 0.6,
        "C_snow_thr": 0.3,
        "C_snow_const": 0.0014,
    }

    bounds = {
        "C_area": (0.0, 10.0),
        "C_fog": (0.0, 1.0),
        "C_dew": (0.0, 1.0),
        "C_rain": (0.1, 1.0),
        "C_rad_tot_k1": (0.0, 1.0),
        "C_rad_tot_k2": (0.0, 1.0),
        "C_rad_tot_k3": (0.0, 1.0),
        "C_rad_tot_b1": (0.0, 0.2),
        "C_rad_tot_b2": (0.2, 0.8),
        "C_snow_thr": (0.2, 0.5),
        "C_snow_const": (0.0, 1.0),
    }

    def __init__(
        self,
        weights: Optional[dict[str, float]] = None,
        bounds: Optional[dict[str, float]] = None,
        loss: Callable[
            [np.ndarray, np.ndarray], float
        ] = sklearn.metrics.mean_absolute_error,
        method="Nelder-Mead",
        baseline_temperature: float = 25.0,
        baseline_efficiency: float = 0.17,
        regressors: dict[str, Optional[str]] = default_regressors,
    ) -> None:
        self.baseline_temperature = baseline_temperature
        self.baseline_efficiency = baseline_efficiency
        if bounds is not None:
            self.bounds = bounds
        if weights is not None:
            self.weights = weights

        self.loss = loss
        self.method = method
        self.regressors = regressors

    @staticmethod
    def from_params(params: list[Parameter], **kwargs) -> "SolarOutputRegression":
        return SolarOutputRegression(
            bounds={p.name: p.bounds for p in params},
            weights={p.name: p.value for p in params},
            **kwargs,
        )

    def set_regressors(self, regressors: dict[str, Optional[str]]):
        self.regressors = regressors

    def _is_foggy(
        self, temperature: np.ndarray, dewpoint: np.ndarray, threshold: float = 2.0
    ) -> np.ndarray:
        """
        Determine if it's foggy based on the closeness of temperature to dewpoint.

        Parameters:
        - temperature (float): The current air temperature.
        - dewpoint (float): The current dewpoint temperature.
        - threshold (float): The temperature difference threshold to determine fog. Default is 2.0 degrees.

        Returns:
        - bool: True if foggy conditions are likely, False otherwise.
        """
        return abs(temperature - dewpoint) < threshold

    def _fog_factor(
        self, parameters: dict[str, float], regressors: pd.DataFrame
    ) -> np.ndarray:
        fog_factor = 1.0
        if "C_fog" in parameters:
            fog_intensity = estimate_fog_intensity(
                regressors["temperature"],
                regressors["dewpoint"],
                regressors["windspeed_10m"],
            )
            fog_factor = np.maximum(0, 1 - parameters["C_fog"] * fog_intensity)
        return fog_factor

    def _rain_factor(
        self, parameters: dict[str, float], regressors: pd.DataFrame
    ) -> np.ndarray:
        rain_factor = 1.0
        if "C_rain" in parameters:
            rain_factor = np.maximum(0, 1 - parameters["C_rain"] * regressors["rain"])
        # elif "C_rain_1d" in parameters:
        #    rain_factor = np.maximum(0, 1 + parameters["C_rain_1d"] * regressors["C_rain_1d"])
        return rain_factor

    def _snow_factor(
        self, parameters: dict[str, float], regressors: pd.DataFrame
    ) -> np.ndarray:
        snow_factor = 1.0
        if "C_snow_thr" in parameters and "C_snow_const" in parameters:
            snow_factor = np.where(
                regressors["snowfall"] > parameters["C_snow_thr"],
                parameters["C_snow_const"],
                1.0,
            )
        elif "C_snow_magnitude" in parameters and "C_snow_eps" in parameters:
            snow_factor = np.where(
                regressors["snowfall"] < 0.5,
                parameters["C_snow_magnitude"]
                / (regressors["snowfall"] + parameters["C_snow_eps"]),
                parameters["C_snow_magnitude"] / (0.5 + parameters["C_snow_eps"]),
            )
            snow_factor[
                regressors["snowfall"]
                < (parameters["C_snow_magnitude"] - parameters["C_snow_eps"])
            ] = 1.0
        return snow_factor

    def _temperature_above_stc_factor(
        self, parameters: dict[str, float], regressors: pd.DataFrame
    ):
        temperature_eff = 1.0
        if "C_temperature" in parameters:
            temperature_eff = np.where(
                regressors["temperature"] > self.baseline_temperature,
                1
                + parameters["C_temperature"]
                * (regressors["temperature"] - self.baseline_temperature),
                1.0,
            )
        return temperature_eff

    def _dew_factor(
        self, parameters: dict[str, float], regressors: pd.DataFrame
    ) -> np.ndarray:
        # Adjust for Dew in the Morning
        dew_factor = 1.0
        if "C_dew" in parameters and "hour" in regressors:
            dew_factor = np.where(
                np.logical_and(regressors["hour"] > 6, regressors["hour"] < 9),
                parameters["C_dew"],
                1,
            )
        return dew_factor

    def _shortwave_contrib(
        self, parameters: dict[str, float], regressors: pd.DataFrame
    ):
        if (
            "C_rad_tot_k1" in parameters
            and "C_rad_tot_k2" in parameters
            and "C_rad_tot_k3" in parameters
        ):
            I = regressors["shortwave_radiation"] / 1000
            thr_1 = parameters["C_rad_tot_b1"]
            thr_2 = parameters["C_rad_tot_b2"]
            k_1 = parameters["C_rad_tot_k1"]
            k_2 = parameters["C_rad_tot_k2"]
            k_3 = parameters["C_rad_tot_k3"]
            return np.where(
                I<=thr_1,
                I*k_1,
                np.where(
                    I >= thr_2,
                    (I-thr_2)*k_3 + thr_2*k_2 + thr_1*k_1,
                    (I-thr_1)*k_2 + thr_1*k_1, 
                )
            )
        elif "C_rad_tot_scale" in parameters:
            return (
                regressors["shortwave_radiation"] / 1000 * parameters["C_rad_tot_scale"]
            )
        else:
            return regressors["shortwave_radiation"] / 1000

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        regressors = {k: X[v] for k, v in self.regressors.items() if v is not None}
        baseline_temperature = self.baseline_temperature
        baseline_efficiency = self.baseline_efficiency
        parameters = self.weights

        # rain impact factor
        rain_factor = self._rain_factor(parameters, regressors)

        # snow impact factor
        snow_factor = self._snow_factor(parameters, regressors)

        # wind impact factor
        wind_factor = 1.0
        if "C_wind" in parameters:
            wind_factor = 1 + parameters["C_wind"] * regressors["windspeed_10m"]

        # Adjust for Fog
        fog_factor = self._fog_factor(parameters, regressors)

        # Adjust for Dew in the Morning
        dew_factor = self._dew_factor(parameters, regressors)

        I_shortwave = self._shortwave_contrib(parameters, regressors)

        # effect of temperature above STC
        temperature_eff = self._temperature_above_stc_factor(parameters, regressors)

        # effect of temperature below STC
        temperature_low_factor = 1.0
        if "C_temperature_low" in parameters:
            temperature_low_factor = 1 + parameters["C_temperature_low"] * np.abs(
                baseline_temperature - regressors["temperature"]
            ) ** (1 / 2)
            temperature_low_factor[temperature_low_factor > 1] = 1.0

        # total effectivity
        eta_total = (
            baseline_efficiency
            * rain_factor
            * snow_factor
            * wind_factor
            * temperature_eff
            * fog_factor
            * dew_factor
        )

        C_area = 4.0
        if "C_area" in parameters:
            C_area = parameters["C_area"]

        return C_area * regressors["installed_capacity"] * I_shortwave * eta_total

    def fit(
        self, X: pd.DataFrame, y: np.ndarray[float]
    ) -> "SolarOutputRegression" | NoReturn:
        def objective_function(params: np.ndarray):
            self.weights = dict(zip(self.weights.keys(), params))
            return self.loss(self.predict(X), y)

        # Minimize the objective function
        self.optim_result: OptimizeResult = minimize(
            objective_function,
            x0=list(self.weights.values()),
            bounds=list(self.bounds.values()),
            method=self.method,
            options={"maxiter": 1e4},
        )

        self.weights = dict(zip(self.weights.keys(), self.optim_result.x))

        self.N = len(X)

        if not self.optim_result.success:
            raise Exception("Optimization failed: " + self.optim_result.message)

        return self

    def get_model(self, order: int):
        assert order == 1
        return self

    @staticmethod
    def create_and_fit(
        X: pd.DataFrame, y: np.ndarray, init_params: dict[str, Any]
    ) -> "SolarOutputRegression":
        model = SolarOutputRegression(**init_params)
        model.fit(X, y)
        return model

    def __str__(self):
        return (
            f"---\n:"
            f"{super().__str__()} with:\n"
            f"Weights: {self.weights}\n"
            f"Score: {self.loss}={self.optim_result.fun if self.optim_result is not None else None}\n"
            "---"
        )


class GroupedSolarOutputRegression(ProductionRegressionBase[Literal[1]]):
    _group_to_model: dict[tuple[str, ...] | str, SolarOutputRegression]

    def __init__(
        self,
        group_columns: Sequence[str] = ("product_type",),
        n_processes: int = 6,
        **regressor_params,
    ) -> None:
        self._group_columns = list(group_columns)
        self._regressor_params = regressor_params
        self._n_processes = n_processes

    def _predict_average(self, df: pd.DataFrame) -> np.ndarray:
        return np.mean(
            [model.predict(df) for model in self._group_to_model.values()], axis=0
        )

    def _predict_group(
        self, group: tuple[str, ...] | str, df: pd.DataFrame
    ) -> np.ndarray:
        if not isinstance(group, tuple):
            group = (group,)

        if group in self._group_to_model:
            return self._group_to_model[group].predict(df)
        else:
            print(f"!!!!!!!!!!!!!!!!!!!!!!!!! No group: {group} => predicting average")
            return self._predict_average(df)

    def set_regressors(self, regressors: dict[str, Optional[str]]):
        for m in self._group_columns.values():
            m.set_regressors(regressors)

    def predict(self, X: pd.DataFrame) -> ndarray:
        X = X.copy()
        X["predictions"] = X.groupby(self._group_columns, group_keys=False).apply(
            lambda df: self._predict_group(df.name, df)
        )
        return X["predictions"].to_numpy()

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        X = X.assign(**{"__target": y})
        group_to_df = {g: df for g, df in X.groupby(self._group_columns)}

        with multiprocessing.Pool(processes=self._n_processes) as pool:
            args = [
                (df, df["__target"], self._regressor_params)
                for _, df in group_to_df.items()
            ]
            models = pool.starmap(SolarOutputRegression.create_and_fit, args)

        self._N = len(X)
        self._group_to_model = dict(zip(group_to_df.keys(), models))

        return self

    def _train_score(self):
        s = 0.0
        for model in self._group_to_model.values():
            s += model.optim_result.fun * model.N / self._N
        return s

    def __str__(self) -> str:
        s = ""
        for g, model in self._group_to_model.items():
            s += "\n"
            s += "************"
            s += f"Model of group={g}: \n{str(model)}"
        s += f"\n************\nScore: {self._train_score()}\n************\n"
        return s

    def get_model(self, order: int):
        assert order == 1
        return self

    @classmethod
    def load(cls, path: Path):
        return joblib.load(path)

    def save(self, path: Path):
        joblib.dump(self, path)