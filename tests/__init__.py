import logging.config
import os
from json import load
from pathlib import Path
from typing import Union


def setup_logger(default_path: Union[str, Path] = "logging.json", default_level=logging.INFO, env_key="LOG_CFG"):
    """Setup logging configuration"""
    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value
    path = Path(path)
    if path.is_file():
        with open(path) as opath:
            config = load(opath)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)


setup_logger(Path("tests") / "logging.json")
