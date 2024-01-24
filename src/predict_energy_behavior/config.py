from dataclasses import dataclass

from pathlib import Path
from typing import Any, Optional

@dataclass
class ConfigDir:
    data: Path
    data_raw: Path
    data_interim: Path
    data_processed: Path

@dataclass
class ConfigDownloadGeoData:
    dir: ConfigDir

@dataclass
class ConfigPrepareStations:
    dir: ConfigDir
    min_weight: float = 0.3
    max_distance: float = 100.0
    on_land_only: bool = True

@dataclass
class ConfigPrepareData:
    dir: ConfigDir
    phase: str

@dataclass
class ConfigSplit:
    val_start_date: str

@dataclass
class ConfigExperiment:
    dir: ConfigDir
    phase: str
    mlflow_tracking_uri: Optional[str]
    exp_name: str
    run_name: str
    split: ConfigSplit
    model: Any

@dataclass
class ConfigInference:
    dir: ConfigDir
    debug: bool
    path_model: str