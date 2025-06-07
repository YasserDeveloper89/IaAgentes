# app/utils.py
import logging
import sys
from typing import Optional

def setup_logging(log_level: int = logging.INFO, log_file: Optional[str] = None) -> None:
    """
    Configura logging con formato detallado y salida a consola y opcionalmente a archivo.
    """
    log_format = (
        "%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s"
    )
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=handlers,
        force=True,
    )

def validate_config_keys(config: dict, required_keys: list) -> None:
    """
    Valida que las claves obligatorias existan en el diccionario de configuración.
    """
    missing = [key for key in required_keys if key not in config]
    if missing:
        raise KeyError(f"Faltan claves obligatorias en config: {missing}")
