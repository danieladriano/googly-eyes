"""App configs"""

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar

import yaml
from dacite import Config, from_dict

T = TypeVar("T")


def load_config_yaml(
    cls: Type[T], yaml_path: Path, root_name: Optional[str] = None
) -> T:
    """Load a config dataclass from a yaml file

    Args:
        cls (Type[T]): Dataclass
        yaml_path (Path): Config file path
        root_name (Optional[str], optional): Root name of yaml file to load the config. Defaults to None.

    Returns:
        T: Config class of the type that is passed by the parameter cls
    """
    with open(yaml_path, "r", encoding="utf8") as file_desc:
        yaml_file_dict: Dict[str, Any] = yaml.safe_load(file_desc)
        if root_name is not None:
            yaml_file_dict = yaml_file_dict[root_name]
        return from_dict(
            data_class=cls, data=yaml_file_dict, config=Config(cast=[Enum])
        )


@dataclass
class CascadeClassifierConfig:
    """Cascade Classifier Config"""

    path: str
    scale_factor: float
    min_neighbors: int


@dataclass
class AppConfig:
    """App Config"""

    face_cascade_classifier: CascadeClassifierConfig
    eyes_cascade_classifier: CascadeClassifierConfig

    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> "AppConfig":
        """Load configuration from a file

        Args:
            config_path (Optional[Path], optional): Config path.
            If not provided, load from ./config.yml. Defaults to None.

        Returns:
            PipelineConfig: Configuration
        """
        if config_path is None:
            config_path = Path(os.getcwd(), "./config.yml")
        return load_config_yaml(cls=cls, yaml_path=config_path, root_name="app_config")
