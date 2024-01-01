from dataclasses import dataclass

@dataclass
class ConfigDir:
    data: str
    data_raw: str
    data_processed: str

@dataclass
class ConfigPrepareData:
    dir: ConfigDir
    phase: str