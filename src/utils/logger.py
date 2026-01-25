"""
Configuration du logging pour le projet.
"""

import logging
import sys
from pathlib import Path

from ..config import LOG_LEVEL, LOG_FORMAT, ROOT_DIR


def setup_logger(
    name: str,
    level: str = LOG_LEVEL,
    log_file: Path | None = None
) -> logging.Logger:
    """
    Configure et retourne un logger.

    Args:
        name: Nom du logger (généralement __name__)
        level: Niveau de log (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Chemin optionnel vers un fichier de log

    Returns:
        Logger configuré
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Handler console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(console_handler)

    # Handler fichier si spécifié
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Récupère ou crée un logger avec la configuration par défaut.

    Args:
        name: Nom du logger

    Returns:
        Logger configuré
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name)
    return logger
