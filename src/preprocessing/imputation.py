"""
Module d'imputation des valeurs manquantes.
Stratégies d'imputation pour différents types de variables.
"""

import logging
from typing import Optional, Literal

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer

logger = logging.getLogger(__name__)


def analyze_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyse détaillée des valeurs manquantes.

    Args:
        df: DataFrame à analyser

    Returns:
        DataFrame avec l'analyse des valeurs manquantes
    """
    missing = pd.DataFrame({
        "missing_count": df.isnull().sum(),
        "missing_pct": (df.isnull().sum() / len(df) * 100).round(2),
        "dtype": df.dtypes
    })

    missing = missing[missing["missing_count"] > 0].sort_values(
        "missing_pct", ascending=False
    )

    logger.info(f"Colonnes avec valeurs manquantes: {len(missing)}")
    return missing


def impute_numeric(
    df: pd.DataFrame,
    columns: list[str],
    strategy: Literal["mean", "median", "most_frequent", "constant"] = "median",
    fill_value: Optional[float] = None
) -> pd.DataFrame:
    """
    Impute les valeurs manquantes pour les colonnes numériques.

    Args:
        df: DataFrame à traiter
        columns: Colonnes à imputer
        strategy: Stratégie d'imputation
        fill_value: Valeur de remplacement si strategy="constant"

    Returns:
        DataFrame avec valeurs imputées
    """
    df = df.copy()

    imputer = SimpleImputer(strategy=strategy, fill_value=fill_value)

    for col in columns:
        if col in df.columns:
            n_missing = df[col].isnull().sum()
            if n_missing > 0:
                df[col] = imputer.fit_transform(df[[col]]).ravel()
                logger.info(f"Colonne {col}: {n_missing} valeurs imputées avec {strategy}")

    return df


def impute_categorical(
    df: pd.DataFrame,
    columns: list[str],
    strategy: Literal["most_frequent", "constant"] = "most_frequent",
    fill_value: str = "Unknown"
) -> pd.DataFrame:
    """
    Impute les valeurs manquantes pour les colonnes catégorielles.

    Args:
        df: DataFrame à traiter
        columns: Colonnes à imputer
        strategy: Stratégie d'imputation
        fill_value: Valeur de remplacement si strategy="constant"

    Returns:
        DataFrame avec valeurs imputées
    """
    df = df.copy()

    for col in columns:
        if col in df.columns:
            n_missing = df[col].isnull().sum()
            if n_missing > 0:
                if strategy == "most_frequent":
                    mode_value = df[col].mode()
                    if len(mode_value) > 0:
                        df[col] = df[col].fillna(mode_value[0])
                        logger.info(f"Colonne {col}: {n_missing} valeurs imputées avec mode ({mode_value[0]})")
                else:
                    df[col] = df[col].fillna(fill_value)
                    logger.info(f"Colonne {col}: {n_missing} valeurs imputées avec '{fill_value}'")

    return df


def impute_knn(
    df: pd.DataFrame,
    columns: list[str],
    n_neighbors: int = 5
) -> pd.DataFrame:
    """
    Impute les valeurs manquantes avec KNN.

    Args:
        df: DataFrame à traiter
        columns: Colonnes à imputer
        n_neighbors: Nombre de voisins pour KNN

    Returns:
        DataFrame avec valeurs imputées
    """
    df = df.copy()

    # Sélectionner uniquement les colonnes numériques pour KNN
    numeric_cols = [col for col in columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]

    if not numeric_cols:
        logger.warning("Aucune colonne numérique à imputer avec KNN")
        return df

    imputer = KNNImputer(n_neighbors=n_neighbors)
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

    logger.info(f"Imputation KNN effectuée sur {len(numeric_cols)} colonnes")
    return df


def create_missing_indicators(
    df: pd.DataFrame,
    columns: Optional[list[str]] = None,
    threshold: float = 0.01
) -> pd.DataFrame:
    """
    Crée des indicateurs binaires pour les valeurs manquantes.

    Args:
        df: DataFrame à traiter
        columns: Colonnes pour lesquelles créer des indicateurs (None = auto)
        threshold: Seuil minimal de valeurs manquantes pour créer un indicateur

    Returns:
        DataFrame avec les indicateurs ajoutés
    """
    df = df.copy()

    if columns is None:
        # Auto-sélection des colonnes avec assez de valeurs manquantes
        missing_pct = df.isnull().sum() / len(df)
        columns = missing_pct[missing_pct >= threshold].index.tolist()

    indicators_created = []
    for col in columns:
        if col in df.columns and df[col].isnull().any():
            indicator_name = f"{col}_missing"
            df[indicator_name] = df[col].isnull().astype(int)
            indicators_created.append(indicator_name)

    if indicators_created:
        logger.info(f"Indicateurs de missing créés: {indicators_created}")

    return df
