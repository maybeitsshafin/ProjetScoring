"""
Script d'exploration et prétraitement des données
Phase 1 - Projet Scoring de Crédit
Auteur: Shafin Hamjah
"""

import logging
import sys
from pathlib import Path

# Ajouter le répertoire racine au path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(ROOT_DIR / 'outputs' / 'reports' / 'preprocessing.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Chemins
RAW_DATA_PATH = ROOT_DIR / 'data' / 'raw' / 'credit_scoring_synth_130k_with_date_defaults.csv'
PROCESSED_DATA_PATH = ROOT_DIR / 'data' / 'processed'
FIGURES_PATH = ROOT_DIR / 'outputs' / 'figures'
REPORTS_PATH = ROOT_DIR / 'outputs' / 'reports'

# Créer les dossiers si nécessaire
PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)
FIGURES_PATH.mkdir(parents=True, exist_ok=True)
REPORTS_PATH.mkdir(parents=True, exist_ok=True)

# Configuration des graphiques
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def load_data() -> pd.DataFrame:
    """Charge les données brutes."""
    logger.info(f"Chargement des données depuis {RAW_DATA_PATH}")
    df = pd.read_csv(RAW_DATA_PATH)
    logger.info(f"Données chargées: {df.shape[0]:,} lignes × {df.shape[1]} colonnes")
    return df


def explore_data(df: pd.DataFrame) -> dict:
    """Exploration approfondie des données."""
    logger.info("=" * 60)
    logger.info("EXPLORATION DES DONNÉES")
    logger.info("=" * 60)

    report = {
        'shape': list(df.shape),
        'memory_mb': float(df.memory_usage(deep=True).sum() / 1024**2),
        'columns': df.columns.tolist(),
        'dtypes': {str(k): int(v) for k, v in df.dtypes.value_counts().to_dict().items()}
    }

    # Types de variables
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    logger.info(f"Variables numériques: {len(numeric_cols)}")
    logger.info(f"Variables catégorielles: {len(categorical_cols)}")

    report['numeric_cols'] = numeric_cols
    report['categorical_cols'] = categorical_cols

    # Variable cible
    target_dist = df['TARGET'].value_counts()
    target_pct = df['TARGET'].value_counts(normalize=True) * 100

    logger.info(f"\nDistribution TARGET:")
    logger.info(f"  0 (Non défaut): {target_dist[0]:,} ({target_pct[0]:.2f}%)")
    logger.info(f"  1 (Défaut): {target_dist[1]:,} ({target_pct[1]:.2f}%)")
    logger.info(f"  Ratio déséquilibre: 1:{target_dist[0]/target_dist[1]:.1f}")

    report['target_distribution'] = {
        'non_default': int(target_dist[0]),
        'default': int(target_dist[1]),
        'default_rate': float(target_pct[1])
    }

    return report


def analyze_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Analyse de la qualité des données."""
    logger.info("=" * 60)
    logger.info("ANALYSE DE QUALITÉ")
    logger.info("=" * 60)

    # Valeurs manquantes
    missing = pd.DataFrame({
        'missing_count': df.isnull().sum(),
        'missing_pct': (df.isnull().sum() / len(df) * 100).round(2)
    })
    missing_cols = missing[missing['missing_count'] > 0]

    logger.info(f"\nValeurs manquantes (NaN): {missing['missing_count'].sum()}")
    if len(missing_cols) > 0:
        logger.info("Colonnes avec valeurs manquantes:")
        for col in missing_cols.index:
            logger.info(f"  {col}: {missing_cols.loc[col, 'missing_count']} ({missing_cols.loc[col, 'missing_pct']}%)")

    # Valeurs vides (chaînes vides)
    empty_counts = {}
    for col in df.select_dtypes(include='object').columns:
        empty = (df[col].astype(str).str.strip() == '').sum()
        if empty > 0:
            empty_counts[col] = empty

    if empty_counts:
        logger.info("\nValeurs vides (chaînes vides):")
        for col, count in empty_counts.items():
            logger.info(f"  {col}: {count} ({count/len(df)*100:.2f}%)")

    # Doublons
    n_dup_full = df.duplicated().sum()
    n_dup_id = df.duplicated(subset=['CLIENT_ID']).sum()
    logger.info(f"\nDoublons complets: {n_dup_full}")
    logger.info(f"Doublons CLIENT_ID: {n_dup_id}")

    # Rapport de qualité complet
    quality_report = pd.DataFrame({
        'dtype': df.dtypes,
        'non_null': df.count(),
        'null_count': df.isnull().sum(),
        'null_pct': (df.isnull().sum() / len(df) * 100).round(2),
        'unique': df.nunique(),
        'unique_pct': (df.nunique() / len(df) * 100).round(2)
    })

    # Ajouter stats pour numériques
    for col in df.select_dtypes(include=[np.number]).columns:
        quality_report.loc[col, 'min'] = df[col].min()
        quality_report.loc[col, 'max'] = df[col].max()
        quality_report.loc[col, 'mean'] = df[col].mean()
        quality_report.loc[col, 'std'] = df[col].std()

    return quality_report


def analyze_outliers(df: pd.DataFrame) -> dict:
    """Analyse des outliers pour les variables numériques."""
    logger.info("=" * 60)
    logger.info("ANALYSE DES OUTLIERS")
    logger.info("=" * 60)

    outliers_report = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    # Exclure certaines colonnes
    exclude_cols = ['CLIENT_ID', 'TARGET']
    analyze_cols = [c for c in numeric_cols if c not in exclude_cols]

    for col in analyze_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        outliers_pct = outliers / len(df) * 100

        if outliers_pct > 1:  # Signaler si > 1%
            outliers_report[col] = {
                'count': int(outliers),
                'pct': round(outliers_pct, 2),
                'lower_bound': round(lower, 2),
                'upper_bound': round(upper, 2),
                'min': round(df[col].min(), 2),
                'max': round(df[col].max(), 2)
            }

    logger.info(f"\nVariables avec outliers significatifs (>1%):")
    for col, info in sorted(outliers_report.items(), key=lambda x: x[1]['pct'], reverse=True)[:15]:
        logger.info(f"  {col}: {info['count']} ({info['pct']}%)")

    return outliers_report


def create_visualizations(df: pd.DataFrame):
    """Crée les visualisations pour l'exploration."""
    logger.info("=" * 60)
    logger.info("CRÉATION DES VISUALISATIONS")
    logger.info("=" * 60)

    # 1. Distribution de TARGET
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#2ecc71', '#e74c3c']
    target_counts = df['TARGET'].value_counts()
    bars = ax.bar(['Non défaut (0)', 'Défaut (1)'], target_counts.values, color=colors)
    for bar, count in zip(bars, target_counts.values):
        pct = count / len(df) * 100
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
                f'{count:,}\n({pct:.1f}%)', ha='center', va='bottom')
    ax.set_ylabel('Nombre de clients')
    ax.set_title('Distribution de la variable cible (TARGET)')
    plt.tight_layout()
    plt.savefig(FIGURES_PATH / '01_target_distribution.png', dpi=150)
    plt.close()
    logger.info("  ✓ 01_target_distribution.png")

    # 2. Distribution de l'âge par TARGET
    fig, ax = plt.subplots(figsize=(10, 6))
    df.groupby('TARGET')['AGE'].plot(kind='hist', alpha=0.6, bins=30, ax=ax, legend=True)
    ax.set_xlabel('Âge')
    ax.set_ylabel('Fréquence')
    ax.set_title('Distribution de l\'âge par statut de défaut')
    ax.legend(['Non défaut', 'Défaut'])
    plt.tight_layout()
    plt.savefig(FIGURES_PATH / '02_age_distribution.png', dpi=150)
    plt.close()
    logger.info("  ✓ 02_age_distribution.png")

    # 3. Analyse temporelle
    df_temp = df.copy()
    df_temp['DATE_MONTH'] = pd.to_datetime(df_temp['DATE_MONTH'])
    monthly = df_temp.groupby('DATE_MONTH').agg({
        'CLIENT_ID': 'count',
        'TARGET': ['sum', 'mean']
    }).reset_index()
    monthly.columns = ['DATE_MONTH', 'nb_clients', 'nb_defauts', 'taux_defaut']
    monthly['taux_defaut'] *= 100

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    axes[0].bar(monthly['DATE_MONTH'], monthly['nb_clients'], color='steelblue', alpha=0.7)
    axes[0].set_ylabel('Nombre de dossiers')
    axes[0].set_title('Volume de dossiers par mois d\'origination')
    axes[0].tick_params(axis='x', rotation=45)

    axes[1].plot(monthly['DATE_MONTH'], monthly['taux_defaut'], marker='o', color='coral', linewidth=2)
    axes[1].set_ylabel('Taux de défaut (%)')
    axes[1].set_title('Évolution du taux de défaut')
    axes[1].set_xlabel('Date')
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(FIGURES_PATH / '03_temporal_analysis.png', dpi=150)
    plt.close()
    logger.info("  ✓ 03_temporal_analysis.png")

    # 4. Valeurs manquantes
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=True)
    if len(missing) > 0:
        fig, ax = plt.subplots(figsize=(10, max(6, len(missing) * 0.4)))
        missing.plot(kind='barh', ax=ax, color='coral')
        ax.set_xlabel('Nombre de valeurs manquantes')
        ax.set_title('Valeurs manquantes par variable')
        for i, (idx, val) in enumerate(missing.items()):
            ax.text(val + 10, i, f'{val/len(df)*100:.1f}%', va='center')
        plt.tight_layout()
        plt.savefig(FIGURES_PATH / '04_missing_values.png', dpi=150)
        plt.close()
        logger.info("  ✓ 04_missing_values.png")

    # 5. Distribution des variables financières clés
    fin_vars = ['LOAN_AMOUNT', 'INCOME_MONTHLY', 'DTI_RATIO', 'CREDIT_SCORE']
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, var in enumerate(fin_vars):
        if var in df.columns:
            ax = axes[i]
            df[var].hist(bins=50, ax=ax, color='steelblue', alpha=0.7, edgecolor='white')
            ax.set_title(f'Distribution de {var}')
            ax.set_xlabel(var)
            ax.set_ylabel('Fréquence')

            # Ajouter stats
            stats_text = f'Moy: {df[var].mean():,.0f}\nMéd: {df[var].median():,.0f}\nÉ-T: {df[var].std():,.0f}'
            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(FIGURES_PATH / '05_financial_distributions.png', dpi=150)
    plt.close()
    logger.info("  ✓ 05_financial_distributions.png")

    # 6. Taux de défaut par variable catégorielle
    cat_vars = ['SEX', 'MARITAL_STATUS', 'EDUCATION_LEVEL', 'EMPLOYMENT_TYPE']
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, var in enumerate(cat_vars):
        if var in df.columns:
            ax = axes[i]
            default_rate = df.groupby(var)['TARGET'].mean() * 100
            default_rate.sort_values().plot(kind='barh', ax=ax, color='coral')
            ax.set_xlabel('Taux de défaut (%)')
            ax.set_title(f'Taux de défaut par {var}')
            ax.axvline(df['TARGET'].mean() * 100, color='black', linestyle='--', label='Moyenne')

    plt.tight_layout()
    plt.savefig(FIGURES_PATH / '06_default_rate_by_category.png', dpi=150)
    plt.close()
    logger.info("  ✓ 06_default_rate_by_category.png")

    # 7. Matrice de corrélation (top variables)
    top_numeric = ['TARGET', 'AGE', 'INCOME_MONTHLY', 'LOAN_AMOUNT', 'DTI_RATIO',
                   'CREDIT_SCORE', 'SENIORITY_YEARS', 'NUM_LATE_PAYMENTS_12M',
                   'UTILIZATION_RATE', 'SAVINGS_BALANCE_M1', 'CHECKING_BALANCE_M1']
    top_numeric = [c for c in top_numeric if c in df.columns]

    fig, ax = plt.subplots(figsize=(12, 10))
    corr_matrix = df[top_numeric].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, ax=ax, square=True, linewidths=0.5)
    ax.set_title('Matrice de corrélation (variables clés)')
    plt.tight_layout()
    plt.savefig(FIGURES_PATH / '07_correlation_matrix.png', dpi=150)
    plt.close()
    logger.info("  ✓ 07_correlation_matrix.png")


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prétraitement complet des données."""
    logger.info("=" * 60)
    logger.info("PRÉTRAITEMENT DES DONNÉES")
    logger.info("=" * 60)

    df_clean = df.copy()
    initial_shape = df_clean.shape

    # 1. Conversion de DATE_MONTH
    logger.info("\n1. Conversion de DATE_MONTH en datetime")
    df_clean['DATE_MONTH'] = pd.to_datetime(df_clean['DATE_MONTH'])

    # 2. Traitement des valeurs vides dans REGION
    logger.info("2. Traitement des valeurs vides")
    for col in df_clean.select_dtypes(include='object').columns:
        # Remplacer chaînes vides par NaN
        mask = df_clean[col].astype(str).str.strip() == ''
        n_empty = mask.sum()
        if n_empty > 0:
            df_clean.loc[mask, col] = np.nan
            logger.info(f"   {col}: {n_empty} chaînes vides converties en NaN")

    # 3. Imputation des valeurs manquantes
    logger.info("3. Imputation des valeurs manquantes")

    # REGION -> 'Inconnu'
    if 'REGION' in df_clean.columns:
        n_missing = df_clean['REGION'].isnull().sum()
        if n_missing > 0:
            df_clean['REGION'] = df_clean['REGION'].fillna('INCONNU')
            logger.info(f"   REGION: {n_missing} valeurs imputées avec 'INCONNU'")

    # 4. Conversion des types
    logger.info("4. Optimisation des types de données")

    # Variables binaires -> int8
    binary_cols = ['TARGET', 'IS_HOMEOWNER', 'PHONE_VERIFIED', 'EMAIL_VERIFIED',
                   'HAS_CAR', 'HAS_PREVIOUS_DEFAULT', 'IS_FOREIGN_WORKER',
                   'INSURANCE_TAKEN', 'HAS_COAPPLICANT', 'MANUAL_REVIEW',
                   'GUARANTOR_PRESENT']
    for col in binary_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype('int8')
    logger.info(f"   {len(binary_cols)} variables binaires converties en int8")

    # Variables catégorielles -> category
    cat_cols = ['SEX', 'MARITAL_STATUS', 'EDUCATION_LEVEL', 'EMPLOYMENT_TYPE',
                'HOUSING_TYPE', 'REGION', 'CHANNEL', 'SEGMENT', 'DEVICE_OS',
                'PAYMENT_METHOD', 'RISK_BAND_INTERNAL']
    for col in cat_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype('category')
    logger.info(f"   {len(cat_cols)} variables catégorielles converties en category")

    # Variables entières -> int32
    int_cols = ['CHILDREN_COUNT', 'LOAN_TERM_MONTHS', 'NB_CREDIT_CARDS',
                'NUM_LATE_PAYMENTS_12M', 'NUM_OPEN_LOANS', 'POS_TXN_COUNT_30D',
                'ATM_TXN_COUNT_30D', 'ONLINE_TXN_COUNT_30D', 'DAYS_SINCE_LAST_PAYMENT',
                'INQUIRIES_6M', 'EMPLOYMENT_GAP_MONTHS', 'BRANCH_VISITS_6M',
                'MOBILE_LOGINS_30D', 'WEB_LOGINS_30D']
    for col in int_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype('int32')
    logger.info(f"   {len(int_cols)} variables entières converties en int32")

    # 5. Création de nouvelles variables utiles
    logger.info("5. Création de variables dérivées")

    # Année et mois d'origination
    df_clean['YEAR'] = df_clean['DATE_MONTH'].dt.year.astype('int16')
    df_clean['MONTH'] = df_clean['DATE_MONTH'].dt.month.astype('int8')
    logger.info("   YEAR et MONTH créées depuis DATE_MONTH")

    # Indicateur de solde négatif
    df_clean['HAS_NEGATIVE_BALANCE'] = (
        (df_clean['CHECKING_BALANCE_M1'] < 0) |
        (df_clean['CHECKING_BALANCE_M2'] < 0) |
        (df_clean['CHECKING_BALANCE_M3'] < 0)
    ).astype('int8')
    logger.info("   HAS_NEGATIVE_BALANCE créée")

    # Évolution des soldes (tendance)
    df_clean['SAVINGS_TREND'] = (
        df_clean['SAVINGS_BALANCE_M1'] - df_clean['SAVINGS_BALANCE_M3']
    )
    df_clean['CHECKING_TREND'] = (
        df_clean['CHECKING_BALANCE_M1'] - df_clean['CHECKING_BALANCE_M3']
    )
    logger.info("   SAVINGS_TREND et CHECKING_TREND créées")

    # 6. Vérification finale
    logger.info("\n6. Vérification finale")
    final_missing = df_clean.isnull().sum().sum()
    final_memory = df_clean.memory_usage(deep=True).sum() / 1024**2

    logger.info(f"   Dimensions: {df_clean.shape[0]:,} × {df_clean.shape[1]}")
    logger.info(f"   Valeurs manquantes restantes: {final_missing}")
    logger.info(f"   Mémoire utilisée: {final_memory:.2f} MB")

    return df_clean


def save_results(df_clean: pd.DataFrame, quality_report: pd.DataFrame,
                 exploration_report: dict, outliers_report: dict):
    """Sauvegarde les résultats."""
    logger.info("=" * 60)
    logger.info("SAUVEGARDE DES RÉSULTATS")
    logger.info("=" * 60)

    # Données nettoyées en Parquet
    output_file = PROCESSED_DATA_PATH / 'credit_data_cleaned.parquet'
    df_clean.to_parquet(output_file, index=False)
    file_size = output_file.stat().st_size / 1024**2
    logger.info(f"  ✓ Données nettoyées: {output_file} ({file_size:.2f} MB)")

    # Rapport de qualité
    quality_file = REPORTS_PATH / 'data_quality_report.csv'
    quality_report.to_csv(quality_file)
    logger.info(f"  ✓ Rapport qualité: {quality_file}")

    # Rapport d'exploration (JSON)
    import json
    exploration_file = REPORTS_PATH / 'exploration_report.json'
    with open(exploration_file, 'w', encoding='utf-8') as f:
        json.dump(exploration_report, f, indent=2, default=str)
    logger.info(f"  ✓ Rapport exploration: {exploration_file}")

    # Rapport outliers
    outliers_file = REPORTS_PATH / 'outliers_report.json'
    with open(outliers_file, 'w', encoding='utf-8') as f:
        json.dump(outliers_report, f, indent=2, default=lambda x: int(x) if hasattr(x, 'item') else str(x))
    logger.info(f"  ✓ Rapport outliers: {outliers_file}")

    # Résumé markdown
    summary_file = REPORTS_PATH / 'preprocessing_summary.md'
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("# Résumé du Prétraitement\n\n")
        f.write(f"**Date d'exécution**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write("## Données\n\n")
        f.write(f"- **Fichier source**: `{RAW_DATA_PATH.name}`\n")
        f.write(f"- **Dimensions finales**: {df_clean.shape[0]:,} lignes × {df_clean.shape[1]} colonnes\n")
        f.write(f"- **Mémoire**: {df_clean.memory_usage(deep=True).sum()/1024**2:.2f} MB\n\n")

        f.write("## Variable cible\n\n")
        f.write(f"- Non défaut (0): {exploration_report['target_distribution']['non_default']:,} ")
        f.write(f"({100 - exploration_report['target_distribution']['default_rate']:.2f}%)\n")
        f.write(f"- Défaut (1): {exploration_report['target_distribution']['default']:,} ")
        f.write(f"({exploration_report['target_distribution']['default_rate']:.2f}%)\n\n")

        f.write("## Actions réalisées\n\n")
        f.write("1. Conversion de DATE_MONTH en datetime\n")
        f.write("2. Imputation des valeurs manquantes (REGION -> 'INCONNU')\n")
        f.write("3. Optimisation des types de données\n")
        f.write("4. Création de variables dérivées (YEAR, MONTH, trends, etc.)\n\n")

        f.write("## Fichiers générés\n\n")
        f.write("- `data/processed/credit_data_cleaned.parquet`\n")
        f.write("- `outputs/reports/data_quality_report.csv`\n")
        f.write("- `outputs/reports/exploration_report.json`\n")
        f.write("- `outputs/reports/outliers_report.json`\n")
        f.write("- `outputs/figures/01-07_*.png`\n")

    logger.info(f"  ✓ Résumé: {summary_file}")


def main():
    """Fonction principale."""
    logger.info("=" * 60)
    logger.info("PHASE 1 - EXPLORATION ET PRÉTRAITEMENT")
    logger.info(f"Début: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    # 1. Charger les données
    df = load_data()

    # 2. Explorer les données
    exploration_report = explore_data(df)

    # 3. Analyser la qualité
    quality_report = analyze_quality(df)

    # 4. Analyser les outliers
    outliers_report = analyze_outliers(df)

    # 5. Créer les visualisations
    create_visualizations(df)

    # 6. Prétraiter les données
    df_clean = preprocess_data(df)

    # 7. Sauvegarder les résultats
    save_results(df_clean, quality_report, exploration_report, outliers_report)

    logger.info("=" * 60)
    logger.info("TERMINÉ AVEC SUCCÈS")
    logger.info(f"Fin: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    return df_clean


if __name__ == "__main__":
    df_clean = main()
