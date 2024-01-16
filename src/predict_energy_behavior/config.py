from dataclasses import dataclass

from pathlib import Path

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
class ConfigExperiment:
    dir: ConfigDir
    phase: str