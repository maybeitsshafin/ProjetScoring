"""
Fonctions d'entrée/sortie pour le projet.
"""

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import joblib

logger = logging.getLogger(__name__)


def load_csv(
    filepath: Path | str,
    **kwargs: Any
) -> pd.DataFrame:
    """
    Charge un fichier CSV en DataFrame.

    Args:
        filepath: Chemin vers le fichier CSV
        **kwargs: Arguments supplémentaires pour pd.read_csv

    Returns:
        DataFrame chargé
    """
    filepath = Path(filepath)
    logger.info(f"Chargement du fichier: {filepath}")

    df = pd.read_csv(filepath, **kwargs)
    logger.info(f"Données chargées: {df.shape[0]} lignes, {df.shape[1]} colonnes")

    return df


def load_parquet(
    filepath: Path | str,
    **kwargs: Any
) -> pd.DataFrame:
    """
    Charge un fichier Parquet en DataFrame.

    Args:
        filepath: Chemin vers le fichier Parquet
        **kwargs: Arguments supplémentaires pour pd.read_parquet

    Returns:
        DataFrame chargé
    """
    filepath = Path(filepath)
    logger.info(f"Chargement du fichier: {filepath}")

    df = pd.read_parquet(filepath, **kwargs)
    logger.info(f"Données chargées: {df.shape[0]} lignes, {df.shape[1]} colonnes")

    return df


def save_parquet(
    df: pd.DataFrame,
    filepath: Path | str,
    **kwargs: Any
) -> None:
    """
    Sauvegarde un DataFrame en format Parquet.

    Args:
        df: DataFrame à sauvegarder
        filepath: Chemin de destination
        **kwargs: Arguments supplémentaires pour to_parquet
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(filepath, **kwargs)
    logger.info(f"Données sauvegardées: {filepath}")


def save_model(
    model: Any,
    filepath: Path | str
) -> None:
    """
    Sauvegarde un modèle avec joblib.

    Args:
        model: Modèle à sauvegarder
        filepath: Chemin de destination
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, filepath)
    logger.info(f"Modèle sauvegardé: {filepath}")


def load_model(filepath: Path | str) -> Any:
    """
    Charge un modèle sauvegardé avec joblib.

    Args:
        filepath: Chemin vers le modèle

    Returns:
        Modèle chargé
    """
    filepath = Path(filepath)
    logger.info(f"Chargement du modèle: {filepath}")

    return joblib.load(filepath)
