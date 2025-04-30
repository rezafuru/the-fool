from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Mapping

import logging


@dataclass
class WandBConfig:
    project_name: str = "N/A"
    entity: str = "N/A"
    team: str = None
    notes: str = None
    tags: List[str] = field(default_factory=list)
    group: str = None
    enabled: bool = False
    resume: bool = False
    id: str = None
    run_name: str = None
    config: Mapping[str, Any] = field(
        default_factory=dict
    )  # The experiment config we log into the experiment


@dataclass
class LoggingConfig:
    # file_log_handler_path : Path = Path("")
    enable_stream_log_handler: bool = True
    enable_file_log_handler: bool = False
    stream_log_handler_level: int = logging.DEBUG
    file_log_handler_level: int = logging.ERROR


@dataclass
class DatasetConfig:
    root: Path = "./"


@dataclass
class DataVizConfig:
    dataset: DatasetConfig
    outputpath: Path
    min_fig_size: int = 10
