from pathlib import Path
from typing import Any
import joblib
from sklearn.ensemble import VotingRegressor
from predict_energy_behavior.models.consumption.base_model import (
    ConsumptionRegressionBase,
)
import lightgbm as lgb

class LGBMSecondOrderModel(ConsumptionRegressionBase):
    def __init__(
        self,
        features: list[str],
        n_models: int,
        parameters: dict[str, Any],
        n_jobs: int = 2,
        n_gpus: int = 4,
    ) -> None:
        model = VotingRegressor(
            estimators=[
                (
                    f"regressor_{i}",
                    lgb.sklearn.LGBMRegressor(
                        **parameters, random_state=i, gpu_device_id=i % n_gpus
                    ),
                )
                for i in range(n_models)
            ],
            n_jobs=n_jobs,
        )
        super().__init__(features=features, model=model)

    @classmethod
    def load(cls, path: Path, name_prefix="consumption_2_") -> "LGBMSecondOrderModel":
        m = joblib.load(path / f"{name_prefix}model_obj.pickle")
        for estimator_name, estimator in m.model.named_estimators_.items():
            model_path = path / f"{name_prefix}{estimator_name}.txt"
            new_booster = lgb.Booster(model_file=model_path)
            if hasattr(estimator, "_Booster"):
                estimator._Booster = new_booster
        return m

    def save(self, path: Path, name_prefix="consumption_2_"):
        joblib.dump(self, path / f"{name_prefix}model_obj.pickle")
        for name, estimator in self.model.named_estimators_.items():
            # Check if the estimator is LGBMRegressor and has been fitted
            if hasattr(estimator, "booster_"):
                model_path = path / f"{name_prefix}{name}.txt"
                estimator.booster_.save_model(model_path)