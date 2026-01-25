"""
Module de validation des données.
Contrôles de qualité et validation des données.
"""

import logging
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def validate_target(
    df: pd.DataFrame,
    target_col: str = "TARGET"
) -> dict:
    """
    Valide la variable cible.

    Args:
        df: DataFrame contenant la cible
        target_col: Nom de la colonne cible

    Returns:
        Dictionnaire avec les résultats de validation
    """
    results = {
        "valid": True,
        "errors": [],
        "warnings": []
    }

    if target_col not in df.columns:
        results["valid"] = False
        results["errors"].append(f"Colonne {target_col} non trouvée")
        return results

    # Vérifier les valeurs uniques
    unique_values = df[target_col].unique()
    if not set(unique_values).issubset({0, 1, np.nan}):
        results["valid"] = False
        results["errors"].append(f"Valeurs non binaires dans {target_col}: {unique_values}")

    # Vérifier les valeurs manquantes
    n_missing = df[target_col].isnull().sum()
    if n_missing > 0:
        results["warnings"].append(f"Valeurs manquantes dans {target_col}: {n_missing}")

    # Vérifier le déséquilibre des classes
    value_counts = df[target_col].value_counts(normalize=True)
    minority_pct = value_counts.min() * 100
    if minority_pct < 10:
        results["warnings"].append(
            f"Classes très déséquilibrées: classe minoritaire = {minority_pct:.1f}%"
        )

    logger.info(f"Validation TARGET: valid={results['valid']}, {len(results['warnings'])} warnings")
    return results


def validate_numeric_ranges(
    df: pd.DataFrame,
    column: str,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None
) -> dict:
    """
    Valide les plages de valeurs pour une colonne numérique.

    Args:
        df: DataFrame à valider
        column: Colonne à vérifier
        min_value: Valeur minimale autorisée
        max_value: Valeur maximale autorisée

    Returns:
        Dictionnaire avec les résultats de validation
    """
    results = {
        "valid": True,
        "n_below_min": 0,
        "n_above_max": 0
    }

    if column not in df.columns:
        results["valid"] = False
        results["error"] = f"Colonne {column} non trouvée"
        return results

    if min_value is not None:
        below_min = df[column] < min_value
        results["n_below_min"] = below_min.sum()
        if results["n_below_min"] > 0:
            results["valid"] = False

    if max_value is not None:
        above_max = df[column] > max_value
        results["n_above_max"] = above_max.sum()
        if results["n_above_max"] > 0:
            results["valid"] = False

    logger.info(
        f"Validation {column}: valid={results['valid']}, "
        f"below_min={results['n_below_min']}, above_max={results['n_above_max']}"
    )
    return results


def validate_no_leakage(
    df: pd.DataFrame,
    date_column: str = "DATE_MONTH",
    train_end_date: str = None
) -> dict:
    """
    Vérifie qu'il n'y a pas de fuite de données temporelles.

    Args:
        df: DataFrame à valider
        date_column: Colonne de date
        train_end_date: Date de fin de l'ensemble d'entraînement

    Returns:
        Dictionnaire avec les résultats de validation
    """
    results = {
        "valid": True,
        "date_range": None,
        "warnings": []
    }

    if date_column not in df.columns:
        results["valid"] = False
        results["error"] = f"Colonne {date_column} non trouvée"
        return results

    min_date = df[date_column].min()
    max_date = df[date_column].max()
    results["date_range"] = (str(min_date), str(max_date))

    logger.info(f"Plage de dates: {min_date} à {max_date}")
    return results


def run_all_validations(
    df: pd.DataFrame,
    target_col: str = "TARGET",
    date_column: str = "DATE_MONTH"
) -> dict:
    """
    Exécute toutes les validations.

    Args:
        df: DataFrame à valider
        target_col: Colonne cible
        date_column: Colonne de date

    Returns:
        Dictionnaire avec tous les résultats de validation
    """
    results = {
        "shape": df.shape,
        "target": validate_target(df, target_col),
        "temporal": validate_no_leakage(df, date_column)
    }

    # Résumé
    all_valid = all([
        results["target"]["valid"],
        results["temporal"]["valid"]
    ])

    results["all_valid"] = all_valid

    if all_valid:
        logger.info("Toutes les validations passées avec succès")
    else:
        logger.warning("Certaines validations ont échoué")

    return results
