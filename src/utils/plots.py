"""
Fonctions de visualisation pour le projet.
"""

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from ..config import FIGURES_DIR

logger = logging.getLogger(__name__)

# Configuration par défaut
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")


def save_figure(
    fig: plt.Figure,
    filename: str,
    output_dir: Path = FIGURES_DIR,
    dpi: int = 150
) -> None:
    """
    Sauvegarde une figure matplotlib.

    Args:
        fig: Figure à sauvegarder
        filename: Nom du fichier (avec extension)
        output_dir: Répertoire de destination
        dpi: Résolution
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filepath = output_dir / filename
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
    logger.info(f"Figure sauvegardée: {filepath}")
    plt.close(fig)


def plot_missing_values(
    df: pd.DataFrame,
    figsize: tuple = (12, 8),
    save: bool = False
) -> plt.Figure:
    """
    Visualise les valeurs manquantes du DataFrame.

    Args:
        df: DataFrame à analyser
        figsize: Taille de la figure
        save: Si True, sauvegarde la figure

    Returns:
        Figure matplotlib
    """
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=True)

    if len(missing) == 0:
        logger.info("Aucune valeur manquante détectée")
        return None

    fig, ax = plt.subplots(figsize=figsize)
    missing.plot(kind="barh", ax=ax, color="coral")
    ax.set_xlabel("Nombre de valeurs manquantes")
    ax.set_title("Valeurs manquantes par variable")

    # Ajouter les pourcentages
    for i, (idx, val) in enumerate(missing.items()):
        pct = val / len(df) * 100
        ax.text(val + 0.5, i, f"{pct:.1f}%", va="center")

    plt.tight_layout()

    if save:
        save_figure(fig, "missing_values.png")

    return fig


def plot_target_distribution(
    df: pd.DataFrame,
    target_col: str = "TARGET",
    figsize: tuple = (8, 6),
    save: bool = False
) -> plt.Figure:
    """
    Visualise la distribution de la variable cible.

    Args:
        df: DataFrame contenant la cible
        target_col: Nom de la colonne cible
        figsize: Taille de la figure
        save: Si True, sauvegarde la figure

    Returns:
        Figure matplotlib
    """
    fig, ax = plt.subplots(figsize=figsize)

    counts = df[target_col].value_counts()
    colors = ["#2ecc71", "#e74c3c"]

    bars = ax.bar(counts.index.astype(str), counts.values, color=colors)
    ax.set_xlabel("Classe")
    ax.set_ylabel("Nombre d'observations")
    ax.set_title(f"Distribution de {target_col}")

    # Ajouter les pourcentages
    total = counts.sum()
    for bar, count in zip(bars, counts.values):
        pct = count / total * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{count:,}\n({pct:.1f}%)",
            ha="center",
            va="bottom"
        )

    plt.tight_layout()

    if save:
        save_figure(fig, "target_distribution.png")

    return fig


def plot_correlation_matrix(
    df: pd.DataFrame,
    figsize: tuple = (14, 12),
    save: bool = False
) -> plt.Figure:
    """
    Visualise la matrice de corrélation.

    Args:
        df: DataFrame avec variables numériques
        figsize: Taille de la figure
        save: Si True, sauvegarde la figure

    Returns:
        Figure matplotlib
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=figsize)

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=False,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax
    )
    ax.set_title("Matrice de corrélation")

    plt.tight_layout()

    if save:
        save_figure(fig, "correlation_matrix.png")

    return fig
