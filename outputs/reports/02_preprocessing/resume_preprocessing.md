# Résumé du Prétraitement

**Date d'exécution**: 2026-01-23 15:49

## Données

- **Fichier source**: `credit_scoring_synth_130k_with_date_defaults.csv`
- **Dimensions finales**: 130,000 lignes × 71 colonnes
- **Mémoire**: 41.91 MB

## Variable cible

- Non défaut (0): 105,820 (81.40%)
- Défaut (1): 24,180 (18.60%)

## Actions réalisées

1. Conversion de DATE_MONTH en datetime
2. Imputation des valeurs manquantes (REGION -> 'INCONNU')
3. Optimisation des types de données
4. Création de variables dérivées (YEAR, MONTH, trends, etc.)

## Fichiers générés

- `data/processed/credit_data_cleaned.parquet`
- `outputs/reports/data_quality_report.csv`
- `outputs/reports/exploration_report.json`
- `outputs/reports/outliers_report.json`
- `outputs/figures/01-07_*.png`
