import logging
import os
from datetime import datetime
from typing import Optional

def get_logger(name: str):
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

def timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def get_api_key() -> Optional[str]:
    return os.getenv("API_KEY") or os.getenv("GEMINI_API_KEY")
