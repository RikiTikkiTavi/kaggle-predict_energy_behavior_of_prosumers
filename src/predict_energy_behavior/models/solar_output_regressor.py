from scipy.optimize import minimize, OptimizeResult
import sklearn.metrics
from typing import Callable, NoReturn, Sequence, Optional
from dataclasses import dataclass
import numpy as np
import pandas as pd

def calculate_relative_humidity(T, D):
    """
    Calculate the relative humidity based on temperature and dewpoint.
    T: Temperature in Celsius
    D: Dewpoint in Celsius
    """
    return 100 * (np.exp((17.625 * D) / (243.04 + D)) / np.exp((17.625 * T) / (243.04 + T)))

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

# Example usage
temperature = 10  # degrees Celsius
dewpoint = 9     # degrees Celsius
windspeed = 1    # m/s

fog_intensity = estimate_fog_intensity(temperature, dewpoint, windspeed)
fog_intensity

@dataclass
class Parameter:
    name: str
    value: float
    bounds: tuple[float, float]
    fixed: bool = False


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
        method="Powell",
        baseline_temperature: float = 25.0,
        baseline_efficiency: float = 0.17,
    ) -> None:
        self.baseline_temperature = baseline_temperature
        self.baseline_efficiency = baseline_efficiency
        self.regressors = regressors
        self.bounds = bounds
        self.weights = weights
        self.loss = loss
        self.method=method
    
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

    def _fog_factor(self, parameters: dict[str, float], regressors: pd.DataFrame) -> np.ndarray:
        fog_factor = 1.0
        if "C_fog" in parameters:
            fog_intensity = estimate_fog_intensity(regressors["temperature"], regressors["dewpoint"], regressors["windspeed_10m"])
            fog_factor = np.maximum(0, 1 - parameters["C_fog"] * fog_intensity)
        return fog_factor

    def _rain_factor(self, parameters: dict[str, float], regressors: pd.DataFrame) -> np.ndarray:
        rain_factor = 1.0
        if "C_rain" in parameters:
            rain_factor = np.maximum(0, 1 - parameters["C_rain"] * regressors["rain"])
        #elif "C_rain_1d" in parameters:
        #    rain_factor = np.maximum(0, 1 + parameters["C_rain_1d"] * regressors["C_rain_1d"])
        return rain_factor

    def _snow_factor(self, parameters: dict[str, float], regressors: pd.DataFrame) -> np.ndarray:
        snow_factor = 1.0
        if "C_snow_3d" in parameters:
            snow_factor = np.maximum(0, 1 - parameters["C_snow_3d"] * regressors["snowfall_3d"]**4)
            # snow_factor = np.where(regressors["snowfall"] > parameters["Thr_snow_cov_100"], 0.0, snow_partial_cov_factor)
        return snow_factor
    
    def _temperature_above_stc_factor(self, parameters: dict[str, float], regressors: pd.DataFrame):
        temperature_eff = 1.0
        if "C_temperature" in parameters:
            temperature_eff = np.where(
                regressors["temperature"] > self.baseline_temperature,
                1 + parameters["C_temperature"] * (regressors["temperature"] - self.baseline_temperature),
                1.0
            )
        return temperature_eff

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        regressors = X
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
        dew_factor = 1.0
        if "C_dew" in parameters and "hour" in regressors:
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
        
        
        I_direct = regressors["direct_solar_radiation"]
        if "C_rad_direct_lin" in parameters and "C_rad_direct_scale" in parameters:
            I_direct = np.where(
                regressors["direct_solar_radiation"] < parameters["C_rad_direct_lin"],
                regressors["direct_solar_radiation"] * parameters["C_rad_direct_scale"],
                regressors["direct_solar_radiation"] * parameters["C_rad_direct_lin"]
            )
        
        I_diffuse = 0.0
        if f"C_rad_diffuse" in parameters:
            I_diffuse += parameters["C_rad_diffuse"] * regressors["diffuse_radiation"]
            
        I_adj = I_adj_total + I_adj_cc + I_direct + I_diffuse

        E = I_adj / 1000

        # effect of temperature above STC
        temperature_eff = self._temperature_above_stc_factor(parameters, regressors)

        # effect of temperature below STC
        temperature_low_factor = 1.0
        if "C_temperature_low" in parameters:
            temperature_low_factor = 1 + parameters["C_temperature_low"] * np.abs(baseline_temperature - regressors["temperature"])**(1/2)
            temperature_low_factor[temperature_low_factor > 1] = 1.0
        
        # total effectivity
        eta_total = baseline_efficiency * rain_factor * snow_factor * wind_factor * temperature_eff * fog_factor * dew_factor

        C_area = 4.0
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
            method=self.method, 
            options={"maxiter": 1e4}
        )

        self.weights = dict(zip(self.weights.keys(), self.optim_result.x)) 

        if not self.optim_result.success:
            raise Exception("Optimization failed: " + self.optim_result.message)

        return self