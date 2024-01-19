import logging
import pickle
import sklearn
from sklearn.ensemble import VotingRegressor
import predict_energy_behavior.config as config
import predict_energy_behavior.models.joined_model as joined_model
import predict_energy_behavior.utils as utils
import pandas as pd
from pathlib import Path
import hydra
import mlflow
import lightgbm

_logger = logging.getLogger(__name__)


@hydra.main(config_path="../../configs", config_name="inference", version_base="1.3")
def main(cfg: config.ConfigInference):
    pass
