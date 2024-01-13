from scipy.optimize import minimize, OptimizeResult
import sklearn.metrics
from typing import Callable, NoReturn, Sequence, Optional
from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass
class Parameter:
    name: str
    value: float
    bounds: tuple[float, float]


class SolarOutputRegressor:

    regressors: Sequence[str]
    weights: dict[str, float]
    bounds: dict[str, tuple[float, float]]
    baseline_temperature: float
    baseline_efficiency: float
    loss: Callable[[np.ndarray, np.ndarray], float]
    optim_result: Optional[OptimizeResult] = None

    def __init__(
        self,
        regressors: Sequence[str],
        weights: dict[str, float],
        bounds: dict[str, float],
        loss: Callable[[np.ndarray, np.ndarray], float] = sklearn.metrics.mean_absolute_error,
        baseline_temperature: float = 25.0,
        baseline_efficiency: float = 0.17,
    ) -> None:
        self.baseline_temperature = baseline_temperature
        self.baseline_efficiency = baseline_efficiency
        self.regressors = regressors
        self.bounds = bounds
        self.weights = weights
        self.loss = loss
    
    @staticmethod
    def from_params(params: list[Parameter], **kwargs) -> "SolarOutputRegressor":
        return SolarOutputRegressor(
            bounds={p.name:p.bounds for p in params},
            weights={p.name:p.value for p in params},
            **kwargs
        )

    def _is_foggy(self, temperature: np.ndarray, dewpoint: np.ndarray, threshold: float = 2.0) -> np.ndarray:
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

    def _rain_factor(self, parameters: dict[str, float], regressors: pd.DataFrame) -> np.ndarray:
        rain_factor = 1.0
        if "C_rain" in parameters:
            rain_factor = np.maximum(0, 1 - parameters["C_rain"] * regressors["rain"])
        elif "C_rain_1d" in parameters:
            rain_factor = np.maximum(0, 1 + parameters["C_rain_1d"] * regressors["C_rain_1d"])
        return rain_factor

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        regressors = X
        baseline_temperature = self.baseline_temperature
        baseline_efficiency = self.baseline_efficiency
        parameters = self.weights

        # rain impact factor
        rain_factor = self._rain_factor(parameters, regressors)
        
        # snow impact factor
        snow_factor = 1.0
        if "C_snow" in parameters:
            snow_partial_cov_factor = np.maximum(0, 1 - parameters["C_snow"] * regressors["snowfall"])
            snow_factor = np.where(regressors["snowfall"] > parameters["Thr_snow_cov_100"], 0.0, snow_partial_cov_factor)

        # wind impact factor
        wind_factor = 1.0
        if "C_wind" in parameters:
            wind_factor = 1 + parameters["C_wind"] * regressors["windspeed_10m"]

        # Adjust for Fog
        fog_factor = 1.0
        if "C_fog" in parameters:
            fog_factor = np.where(
                self._is_foggy(regressors["dewpoint"], regressors["temperature"], parameters["Thr_foggy"]),
                parameters["C_fog"],
                1 
            )

        # Adjust for Dew in the Morning
        dew_factor = 1.0
        if "C_dew" in parameters:
            dew_factor = np.where(
                np.logical_and(regressors["hour"] > 6, regressors["hour"] < 9),
                parameters["C_dew"],
                1
            )

        I_adj_total = 0.0
        # adjusted total horizontal irradiance
        if "shortwave_radiation" in regressors and "C_cloud_total" in parameters:
            I_adj_total = regressors["shortwave_radiation"] * (1 - parameters["C_cloud_total"] * regressors['cloudcover_total'] / 100)
        
        # Direct and diffuse radiation adjusted to cloud cover
        I_adj_cc = 0.0
        rad_to_reg = {"direct": "direct_solar_radiation", "diffuse": "diffuse_radiation"}
        for rad in ["direct", "diffuse"]:
            for cc in ["total", "high", "mid", "low"]:
                if f"C_CC_{cc}_{rad}" in parameters:
                    I_adj_cc += regressors[rad_to_reg[rad]] * (1 - parameters[f"C_CC_{cc}_{rad}"] * regressors[f'cloudcover_{cc}'] / 100) 
        
        I_adj = I_adj_total + I_adj_cc

        E = I_adj / 1000

        # effect of temperature above STC
        temperature_eff = 1.0
        if "C_temperature" in parameters:
            temperature_eff = 1 + parameters["C_temperature"] * (regressors["temperature"] - baseline_temperature)
            temperature_eff[temperature_eff < 1] = 1


        # effect of temperature below STC
        temperature_low_factor = 1.0
        if "C_temperature_low" in parameters:
            temperature_low_factor = 1 + parameters["C_temperature_low"] * np.abs(baseline_temperature - regressors["temperature"])**(1/2)
            temperature_low_factor[temperature_low_factor > 1] = 1.0
        
        # total effectivity
        eta_total = baseline_efficiency * rain_factor * snow_factor * wind_factor * temperature_eff * fog_factor * dew_factor

        C_area = 1.0
        if "C_area" in parameters:
            C_area = parameters["C_area"]

        return C_area * regressors["installed_capacity"] * E * eta_total
    

    def fit(self, X: pd.DataFrame, y: np.ndarray[float]) -> "SolarOutputRegressor" | NoReturn:
        def objective_function(params: np.ndarray):
            self.weights = dict(zip(self.weights.keys(), params))
            return self.loss(self.predict(X), y)

        # Minimize the objective function
        self.optim_result: OptimizeResult = minimize(
            objective_function, 
            x0=list(self.weights.values()), 
            bounds=list(self.bounds.values()), 
            method='Nelder-Mead', 
            options={"maxiter": 1e4}
        )

        self.weights = dict(zip(self.weights.keys(), self.optim_result.x)) 

        if not self.optim_result.success:
            raise Exception("Optimization failed: " + self.optim_result.message)

        return self