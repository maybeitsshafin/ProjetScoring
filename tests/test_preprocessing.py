"""Tests pour le module preprocessing."""

import pytest
import pandas as pd
import numpy as np

from src.preprocessing.cleaning import (
    remove_duplicates,
    standardize_column_names,
    detect_outliers_iqr
)


class TestCleaning:
    """Tests pour les fonctions de nettoyage."""

    def test_remove_duplicates(self):
        """Test de suppression des doublons."""
        df = pd.DataFrame({
            "a": [1, 1, 2, 3],
            "b": ["x", "x", "y", "z"]
        })
        result = remove_duplicates(df)
        assert len(result) == 3

    def test_standardize_column_names(self):
        """Test de standardisation des noms de colonnes."""
        df = pd.DataFrame({
            "Column Name": [1],
            "Another-Column": [2],
            "  Spaces  ": [3]
        })
        result = standardize_column_names(df)
        assert "column_name" in result.columns
        assert "another_column" in result.columns
        assert "spaces" in result.columns

    def test_detect_outliers_iqr(self):
        """Test de détection des outliers."""
        df = pd.DataFrame({
            "values": [1, 2, 3, 4, 5, 100]  # 100 est un outlier
        })
        outliers = detect_outliers_iqr(df, "values")
        assert outliers.sum() >= 1
