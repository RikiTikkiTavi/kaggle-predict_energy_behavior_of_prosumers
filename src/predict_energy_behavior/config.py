from dataclasses import dataclass

from pathlib import Path
from typing import Any

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
class ConfigPrepareData:
    dir: ConfigDir
    phase: str

@dataclass
class ConfigSplit:
    val_start_date: str

@dataclass
class ConfigModel:
    first_order_model: Any
    second_order_model: Any

@dataclass
class ConfigExperiment:
    dir: ConfigDir
    phase: str
    split: ConfigSplit
    model: ConfigModel