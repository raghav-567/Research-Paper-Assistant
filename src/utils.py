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

def get_api_key(preferred_key: Optional[str] = None) -> Optional[str]:
    if preferred_key and preferred_key.strip():
        return preferred_key.strip()
    return os.getenv("API_KEY") or os.getenv("GEMINI_API_KEY")


def mask_api_key(key: Optional[str]) -> str:
    if not key:
        return ""
    key = key.strip()
    if len(key) <= 8:
        return "*" * len(key)
    return f"{key[:4]}{'*' * (len(key) - 8)}{key[-4:]}"
