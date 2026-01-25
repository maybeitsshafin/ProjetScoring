"""
Module de nettoyage des données.
Fonctions pour le nettoyage et la préparation initiale des données.
"""

import logging
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def remove_duplicates(
    df: pd.DataFrame,
    subset: Optional[list[str]] = None,
    keep: str = "first"
) -> pd.DataFrame:
    """
    Supprime les lignes dupliquées du DataFrame.

    Args:
        df: DataFrame à nettoyer
        subset: Colonnes à considérer pour la détection des doublons
        keep: 'first', 'last' ou False

    Returns:
        DataFrame sans doublons
    """
    n_before = len(df)
    df_clean = df.drop_duplicates(subset=subset, keep=keep)
    n_removed = n_before - len(df_clean)

    if n_removed > 0:
        logger.info(f"Doublons supprimés: {n_removed} lignes ({n_removed/n_before*100:.2f}%)")
    else:
        logger.info("Aucun doublon détecté")

    return df_clean


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise les noms de colonnes (minuscules, underscores).

    Args:
        df: DataFrame à traiter

    Returns:
        DataFrame avec noms de colonnes standardisés
    """
    df = df.copy()
    df.columns = (
        df.columns
        .str.lower()
        .str.strip()
        .str.replace(r"[^\w]", "_", regex=True)
        .str.replace(r"_+", "_", regex=True)
        .str.strip("_")
    )
    logger.info("Noms de colonnes standardisés")
    return df


def convert_dtypes(
    df: pd.DataFrame,
    date_columns: Optional[list[str]] = None,
    category_columns: Optional[list[str]] = None
) -> pd.DataFrame:
    """
    Convertit les types de données des colonnes.

    Args:
        df: DataFrame à traiter
        date_columns: Colonnes à convertir en datetime
        category_columns: Colonnes à convertir en category

    Returns:
        DataFrame avec types convertis
    """
    df = df.copy()

    if date_columns:
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
                logger.info(f"Colonne {col} convertie en datetime")

    if category_columns:
        for col in category_columns:
            if col in df.columns:
                df[col] = df[col].astype("category")
                logger.info(f"Colonne {col} convertie en category")

    return df


def detect_outliers_iqr(
    df: pd.DataFrame,
    column: str,
    multiplier: float = 1.5
) -> pd.Series:
    """
    Détecte les outliers avec la méthode IQR.

    Args:
        df: DataFrame contenant la colonne
        column: Nom de la colonne à analyser
        multiplier: Multiplicateur pour les bornes (1.5 par défaut)

    Returns:
        Série booléenne indiquant les outliers
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR

    outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
    n_outliers = outliers.sum()

    logger.info(
        f"Outliers détectés pour {column}: {n_outliers} "
        f"({n_outliers/len(df)*100:.2f}%) - Bornes: [{lower_bound:.2f}, {upper_bound:.2f}]"
    )

    return outliers


def cap_outliers(
    df: pd.DataFrame,
    column: str,
    lower_percentile: float = 0.01,
    upper_percentile: float = 0.99
) -> pd.DataFrame:
    """
    Plafonne les outliers aux percentiles spécifiés.

    Args:
        df: DataFrame à traiter
        column: Colonne à plafonner
        lower_percentile: Percentile inférieur
        upper_percentile: Percentile supérieur

    Returns:
        DataFrame avec outliers plafonnés
    """
    df = df.copy()

    lower_bound = df[column].quantile(lower_percentile)
    upper_bound = df[column].quantile(upper_percentile)

    n_lower = (df[column] < lower_bound).sum()
    n_upper = (df[column] > upper_bound).sum()

    df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)

    logger.info(
        f"Outliers plafonnés pour {column}: {n_lower} (bas) + {n_upper} (haut) "
        f"aux bornes [{lower_bound:.2f}, {upper_bound:.2f}]"
    )

    return df


def get_data_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Génère un rapport de qualité des données.

    Args:
        df: DataFrame à analyser

    Returns:
        DataFrame avec le rapport de qualité
    """
    report = pd.DataFrame({
        "dtype": df.dtypes,
        "non_null": df.count(),
        "null_count": df.isnull().sum(),
        "null_pct": (df.isnull().sum() / len(df) * 100).round(2),
        "unique": df.nunique(),
        "unique_pct": (df.nunique() / len(df) * 100).round(2)
    })

    # Ajouter les statistiques pour les colonnes numériques
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        report.loc[col, "min"] = df[col].min()
        report.loc[col, "max"] = df[col].max()
        report.loc[col, "mean"] = df[col].mean()
        report.loc[col, "std"] = df[col].std()

    logger.info("Rapport de qualité des données généré")
    return report
