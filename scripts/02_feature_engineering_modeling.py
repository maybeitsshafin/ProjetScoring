"""
Script de Feature Engineering et Modélisation Baseline
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
import json
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    classification_report, confusion_matrix, f1_score,
    precision_score, recall_score, accuracy_score
)
from imblearn.over_sampling import SMOTE
import joblib

# Configuration du logging
log_file = ROOT_DIR / 'outputs' / 'reports' / 'phase1_modeling.log'
log_file.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file, encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Chemins
PROCESSED_DATA_PATH = ROOT_DIR / 'data' / 'processed' / 'credit_data_cleaned.parquet'
FIGURES_PATH = ROOT_DIR / 'outputs' / 'figures'
REPORTS_PATH = ROOT_DIR / 'outputs' / 'reports'
MODELS_PATH = ROOT_DIR / 'outputs' / 'models'

# Créer les dossiers
MODELS_PATH.mkdir(parents=True, exist_ok=True)

# Configuration graphiques
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

# Seed pour reproductibilité
RANDOM_STATE = 42


def load_cleaned_data() -> pd.DataFrame:
    """Charge les données nettoyées."""
    logger.info(f"Chargement des données depuis {PROCESSED_DATA_PATH}")
    df = pd.read_parquet(PROCESSED_DATA_PATH)
    logger.info(f"Données chargées: {df.shape[0]:,} lignes × {df.shape[1]} colonnes")
    return df


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def create_financial_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crée des features financières."""
    logger.info("Création des features financières...")

    # Ratio épargne/revenu
    df['SAVINGS_TO_INCOME'] = df['SAVINGS_BALANCE_M1'] / (df['INCOME_MONTHLY'] + 1)

    # Ratio compte courant/revenu
    df['CHECKING_TO_INCOME'] = df['CHECKING_BALANCE_M1'] / (df['INCOME_MONTHLY'] + 1)

    # Total des soldes
    df['TOTAL_BALANCE_M1'] = df['SAVINGS_BALANCE_M1'] + df['CHECKING_BALANCE_M1']
    df['TOTAL_BALANCE_M2'] = df['SAVINGS_BALANCE_M2'] + df['CHECKING_BALANCE_M2']
    df['TOTAL_BALANCE_M3'] = df['SAVINGS_BALANCE_M3'] + df['CHECKING_BALANCE_M3']

    # Moyenne des soldes totaux
    df['AVG_TOTAL_BALANCE'] = (df['TOTAL_BALANCE_M1'] + df['TOTAL_BALANCE_M2'] + df['TOTAL_BALANCE_M3']) / 3

    # Ratio prêt/revenu annuel
    df['LOAN_TO_ANNUAL_INCOME'] = df['LOAN_AMOUNT'] / (df['INCOME_MONTHLY'] * 12 + 1)

    # Capacité de remboursement (revenu - dépenses - mensualité)
    df['REPAYMENT_CAPACITY'] = df['INCOME_MONTHLY'] - df['EXPENSES_MONTHLY'] - df['MONTHLY_INSTALLMENT']

    # Ratio dépenses/revenu
    df['EXPENSE_RATIO'] = df['EXPENSES_MONTHLY'] / (df['INCOME_MONTHLY'] + 1)

    # Marge financière
    df['FINANCIAL_MARGIN'] = df['INCOME_MONTHLY'] - df['EXPENSES_MONTHLY']

    # Indicateur de stress financier (DTI > 0.4 ou EXPENSE_RATIO > 0.7)
    df['FINANCIAL_STRESS'] = ((df['DTI_RATIO'] > 0.4) | (df['EXPENSE_RATIO'] > 0.7)).astype('int8')

    logger.info(f"  → 11 features financières créées")
    return df


def create_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crée des features comportementales."""
    logger.info("Création des features comportementales...")

    # Total des transactions
    df['TOTAL_TXN_30D'] = df['POS_TXN_COUNT_30D'] + df['ATM_TXN_COUNT_30D'] + df['ONLINE_TXN_COUNT_30D']

    # Ratio transactions en ligne
    df['ONLINE_TXN_RATIO'] = df['ONLINE_TXN_COUNT_30D'] / (df['TOTAL_TXN_30D'] + 1)

    # Total connexions digitales
    df['TOTAL_DIGITAL_LOGINS'] = df['MOBILE_LOGINS_30D'] + df['WEB_LOGINS_30D']

    # Ratio mobile/web
    df['MOBILE_PREFERENCE'] = df['MOBILE_LOGINS_30D'] / (df['TOTAL_DIGITAL_LOGINS'] + 1)

    # Score d'engagement digital
    df['DIGITAL_ENGAGEMENT'] = (
        df['TOTAL_DIGITAL_LOGINS'] +
        df['ONLINE_TXN_COUNT_30D'] * 2
    )

    # Client actif en agence
    df['IS_BRANCH_ACTIVE'] = (df['BRANCH_VISITS_6M'] >= 2).astype('int8')

    # Montant moyen retrait/dépôt
    df['WITHDRAWAL_DEPOSIT_RATIO'] = df['LAST_WITHDRAWAL_AMOUNT'] / (df['LAST_DEPOSIT_AMOUNT'] + 1)

    logger.info(f"  → 7 features comportementales créées")
    return df


def create_stability_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crée des features de stabilité."""
    logger.info("Création des features de stabilité...")

    # Stabilité des soldes épargne (écart-type sur 3 mois)
    savings_cols = ['SAVINGS_BALANCE_M1', 'SAVINGS_BALANCE_M2', 'SAVINGS_BALANCE_M3']
    df['SAVINGS_STABILITY'] = df[savings_cols].std(axis=1)

    # Stabilité des soldes courant
    checking_cols = ['CHECKING_BALANCE_M1', 'CHECKING_BALANCE_M2', 'CHECKING_BALANCE_M3']
    df['CHECKING_STABILITY'] = df[checking_cols].std(axis=1)

    # Coefficient de variation des soldes
    df['BALANCE_CV'] = df['SAVINGS_STABILITY'] / (df[savings_cols].mean(axis=1).abs() + 1)

    # Tendance épargne (croissante = positif)
    df['SAVINGS_TREND'] = df['SAVINGS_BALANCE_M1'] - df['SAVINGS_BALANCE_M3']
    df['SAVINGS_TREND_RATIO'] = df['SAVINGS_TREND'] / (df['SAVINGS_BALANCE_M3'].abs() + 1)

    # Tendance compte courant
    df['CHECKING_TREND'] = df['CHECKING_BALANCE_M1'] - df['CHECKING_BALANCE_M3']
    df['CHECKING_TREND_RATIO'] = df['CHECKING_TREND'] / (df['CHECKING_BALANCE_M3'].abs() + 1)

    # Indicateur de stabilité emploi
    df['EMPLOYMENT_STABILITY'] = (df['SENIORITY_YEARS'] >= 2).astype('int8')

    # Stabilité résidence
    df['RESIDENCE_STABILITY'] = (df['RESIDENCE_SINCE_YEARS'] >= 2).astype('int8')

    logger.info(f"  → 7 features de stabilité créées")
    return df


def create_risk_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crée des features de risque."""
    logger.info("Création des features de risque...")

    # Score de risque composite
    df['RISK_SCORE_CUSTOM'] = (
        df['NUM_LATE_PAYMENTS_12M'] * 10 +
        df['HAS_PREVIOUS_DEFAULT'] * 20 +
        df['INQUIRIES_6M'] * 5 +
        (df['UTILIZATION_RATE'] > 0.7).astype(int) * 15 +
        (df['DTI_RATIO'] > 0.5).astype(int) * 10
    )

    # Credit score normalisé (entre 0 et 1)
    df['CREDIT_SCORE_NORM'] = (df['CREDIT_SCORE'] - df['CREDIT_SCORE'].min()) / (df['CREDIT_SCORE'].max() - df['CREDIT_SCORE'].min())

    # Catégorie de credit score
    df['CREDIT_SCORE_CAT'] = pd.cut(
        df['CREDIT_SCORE'],
        bins=[0, 500, 600, 700, 800, 1000],
        labels=['very_poor', 'poor', 'fair', 'good', 'excellent']
    )

    # Ratio prêts ouverts / limite crédit
    df['LOAN_UTILIZATION'] = df['NUM_OPEN_LOANS'] / (df['TOTAL_CREDIT_LIMIT'] / 10000 + 1)

    # Client à haut risque (multiple facteurs)
    df['HIGH_RISK_FLAG'] = (
        (df['NUM_LATE_PAYMENTS_12M'] >= 2) |
        (df['HAS_PREVIOUS_DEFAULT'] == 1) |
        (df['CREDIT_SCORE'] < 550) |
        (df['DTI_RATIO'] > 0.6)
    ).astype('int8')

    # Client nouveau (peu d'historique)
    df['IS_NEW_CLIENT'] = (
        (df['SENIORITY_YEARS'] < 1) &
        (df['NUM_OPEN_LOANS'] == 0)
    ).astype('int8')

    logger.info(f"  → 6 features de risque créées")
    return df


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crée des features d'interaction."""
    logger.info("Création des features d'interaction...")

    # Age × Ancienneté
    df['AGE_SENIORITY'] = df['AGE'] * df['SENIORITY_YEARS']

    # Revenu × Score crédit
    df['INCOME_CREDIT_SCORE'] = df['INCOME_MONTHLY'] * df['CREDIT_SCORE_NORM']

    # DTI × Utilisation crédit
    df['DTI_UTILIZATION'] = df['DTI_RATIO'] * df['UTILIZATION_RATE']

    # Prêt × Taux d'intérêt
    df['LOAN_INTEREST_COST'] = df['LOAN_AMOUNT'] * df['INTEREST_RATE']

    # Balance × Durée du prêt
    df['BALANCE_LOAN_TERM'] = df['AVG_TOTAL_BALANCE'] / (df['LOAN_TERM_MONTHS'] + 1)

    logger.info(f"  → 5 features d'interaction créées")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline complet de feature engineering."""
    logger.info("=" * 60)
    logger.info("FEATURE ENGINEERING")
    logger.info("=" * 60)

    initial_cols = len(df.columns)

    df = create_financial_features(df)
    df = create_behavioral_features(df)
    df = create_stability_features(df)
    df = create_risk_features(df)
    df = create_interaction_features(df)

    # Remplacer les valeurs infinies
    df = df.replace([np.inf, -np.inf], np.nan)

    # Remplir les NaN créés par les divisions
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    new_cols = len(df.columns) - initial_cols
    logger.info(f"\nTotal: {new_cols} nouvelles features créées")
    logger.info(f"Dimensions finales: {df.shape[0]:,} × {df.shape[1]}")

    return df


# =============================================================================
# PRÉPARATION POUR LA MODÉLISATION
# =============================================================================

def prepare_modeling_data(df: pd.DataFrame) -> tuple:
    """Prépare les données pour la modélisation avec split temporel."""
    logger.info("=" * 60)
    logger.info("PRÉPARATION DES DONNÉES POUR MODÉLISATION")
    logger.info("=" * 60)

    # Colonnes à exclure
    exclude_cols = [
        'CLIENT_ID', 'TARGET', 'DATE_MONTH', 'DEFAULTS_ORIGINATION',
        'CREDIT_SCORE_CAT'  # Variable catégorielle créée
    ]

    # Variables catégorielles à encoder
    cat_cols = df.select_dtypes(include=['category']).columns.tolist()
    cat_cols = [c for c in cat_cols if c not in exclude_cols]

    logger.info(f"Variables catégorielles à encoder: {len(cat_cols)}")

    # Encodage des variables catégorielles
    df_encoded = df.copy()
    label_encoders = {}

    for col in cat_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        label_encoders[col] = le

    # Sélection des features
    feature_cols = [c for c in df_encoded.columns if c not in exclude_cols]

    X = df_encoded[feature_cols]
    y = df_encoded['TARGET']

    logger.info(f"Features: {X.shape[1]}")
    logger.info(f"Observations: {X.shape[0]:,}")

    # Split temporel (basé sur DATE_MONTH)
    # Train: avant 2025-01, Test: 2025-01 et après
    train_mask = df['DATE_MONTH'] < '2025-01-01'
    test_mask = df['DATE_MONTH'] >= '2025-01-01'

    X_train = X[train_mask]
    X_test = X[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]

    logger.info(f"\nSplit temporel:")
    logger.info(f"  Train: {len(X_train):,} ({len(X_train)/len(X)*100:.1f}%) - avant 2025-01")
    logger.info(f"  Test:  {len(X_test):,} ({len(X_test)/len(X)*100:.1f}%) - 2025-01 et après")
    logger.info(f"\nTaux de défaut:")
    logger.info(f"  Train: {y_train.mean()*100:.2f}%")
    logger.info(f"  Test:  {y_test.mean()*100:.2f}%")

    return X_train, X_test, y_train, y_test, feature_cols, label_encoders


# =============================================================================
# MODÉLISATION
# =============================================================================

def train_logistic_regression(X_train, X_test, y_train, y_test, feature_cols):
    """Entraîne un modèle de régression logistique."""
    logger.info("=" * 60)
    logger.info("MODÉLISATION - RÉGRESSION LOGISTIQUE (BASELINE)")
    logger.info("=" * 60)

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # SMOTE pour gérer le déséquilibre
    logger.info("\nApplication de SMOTE pour rééquilibrage...")
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    logger.info(f"  Avant SMOTE: {len(y_train):,} (défaut: {y_train.sum():,})")
    logger.info(f"  Après SMOTE: {len(y_train_resampled):,} (défaut: {y_train_resampled.sum():,})")

    # Entraînement
    logger.info("\nEntraînement du modèle...")
    model = LogisticRegression(
        C=1.0,
        class_weight='balanced',
        max_iter=1000,
        random_state=RANDOM_STATE,
        solver='lbfgs'
    )
    model.fit(X_train_resampled, y_train_resampled)

    # Prédictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

    # Métriques
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auc_roc': roc_auc_score(y_test, y_pred_proba)
    }

    logger.info("\n" + "=" * 40)
    logger.info("RÉSULTATS SUR LE JEU DE TEST")
    logger.info("=" * 40)
    logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall:    {metrics['recall']:.4f}")
    logger.info(f"  F1-Score:  {metrics['f1']:.4f}")
    logger.info(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")

    # Rapport de classification
    logger.info("\nRapport de classification:")
    logger.info("\n" + classification_report(y_test, y_pred, target_names=['Non défaut', 'Défaut']))

    # Validation croisée sur le train
    logger.info("\nValidation croisée (5-fold) sur le train...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='roc_auc')
    logger.info(f"  AUC-ROC CV: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

    return model, scaler, metrics, y_test, y_pred, y_pred_proba


def plot_model_results(y_test, y_pred, y_pred_proba, metrics, feature_cols, model):
    """Génère les visualisations des résultats du modèle."""
    logger.info("\nGénération des visualisations...")

    # 1. Courbe ROC
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    axes[0, 0].plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {metrics["auc_roc"]:.3f})')
    axes[0, 0].plot([0, 1], [0, 1], 'k--', linewidth=1)
    axes[0, 0].fill_between(fpr, tpr, alpha=0.3)
    axes[0, 0].set_xlabel('Taux de Faux Positifs')
    axes[0, 0].set_ylabel('Taux de Vrais Positifs')
    axes[0, 0].set_title('Courbe ROC')
    axes[0, 0].legend(loc='lower right')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1],
                xticklabels=['Non défaut', 'Défaut'],
                yticklabels=['Non défaut', 'Défaut'])
    axes[0, 1].set_xlabel('Prédit')
    axes[0, 1].set_ylabel('Réel')
    axes[0, 1].set_title('Matrice de Confusion')

    # 3. Distribution des probabilités
    axes[1, 0].hist(y_pred_proba[y_test == 0], bins=50, alpha=0.7, label='Non défaut', color='green')
    axes[1, 0].hist(y_pred_proba[y_test == 1], bins=50, alpha=0.7, label='Défaut', color='red')
    axes[1, 0].set_xlabel('Probabilité de défaut')
    axes[1, 0].set_ylabel('Fréquence')
    axes[1, 0].set_title('Distribution des probabilités prédites')
    axes[1, 0].legend()
    axes[1, 0].axvline(0.5, color='black', linestyle='--', label='Seuil 0.5')

    # 4. Courbe Precision-Recall
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    axes[1, 1].plot(recall, precision, 'b-', linewidth=2)
    axes[1, 1].fill_between(recall, precision, alpha=0.3)
    axes[1, 1].set_xlabel('Recall')
    axes[1, 1].set_ylabel('Precision')
    axes[1, 1].set_title('Courbe Precision-Recall')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_PATH / '08_model_performance.png', dpi=150)
    plt.close()
    logger.info("  ✓ 08_model_performance.png")

    # 5. Feature Importance (top 20)
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': np.abs(model.coef_[0])
    }).sort_values('importance', ascending=True).tail(20)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.barh(feature_importance['feature'], feature_importance['importance'], color='steelblue')
    ax.set_xlabel('Importance (valeur absolue du coefficient)')
    ax.set_title('Top 20 Features - Régression Logistique')
    plt.tight_layout()
    plt.savefig(FIGURES_PATH / '09_feature_importance.png', dpi=150)
    plt.close()
    logger.info("  ✓ 09_feature_importance.png")

    return feature_importance


# =============================================================================
# ANALYSE TEMPORELLE
# =============================================================================

def analyze_time_series(df: pd.DataFrame):
    """Analyse préliminaire de la série temporelle des défauts."""
    logger.info("=" * 60)
    logger.info("ANALYSE TEMPORELLE PRÉLIMINAIRE")
    logger.info("=" * 60)

    # Agrégation mensuelle
    monthly_data = df.groupby('DATE_MONTH').agg({
        'TARGET': ['count', 'sum', 'mean'],
        'DEFAULTS_ORIGINATION': 'first'
    }).reset_index()
    monthly_data.columns = ['DATE_MONTH', 'nb_dossiers', 'nb_defauts', 'taux_defaut', 'defaults_origination']
    monthly_data['taux_defaut'] *= 100

    logger.info(f"\nPériode analysée: {monthly_data['DATE_MONTH'].min()} à {monthly_data['DATE_MONTH'].max()}")
    logger.info(f"Nombre de mois: {len(monthly_data)}")

    # Statistiques
    logger.info(f"\nStatistiques DEFAULTS_ORIGINATION:")
    logger.info(f"  Moyenne: {monthly_data['defaults_origination'].mean():.1f}")
    logger.info(f"  Écart-type: {monthly_data['defaults_origination'].std():.1f}")
    logger.info(f"  Min: {monthly_data['defaults_origination'].min()}")
    logger.info(f"  Max: {monthly_data['defaults_origination'].max()}")

    # Visualisation
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # 1. Volume de défauts par mois
    axes[0].bar(monthly_data['DATE_MONTH'], monthly_data['defaults_origination'],
                color='coral', alpha=0.8)
    axes[0].set_ylabel('Nombre de défauts')
    axes[0].set_title('Volume de défauts par mois (DEFAULTS_ORIGINATION)')
    axes[0].tick_params(axis='x', rotation=45)

    # 2. Taux de défaut
    axes[1].plot(monthly_data['DATE_MONTH'], monthly_data['taux_defaut'],
                 marker='o', color='steelblue', linewidth=2)
    axes[1].axhline(monthly_data['taux_defaut'].mean(), color='red', linestyle='--',
                    label=f'Moyenne: {monthly_data["taux_defaut"].mean():.2f}%')
    axes[1].set_ylabel('Taux de défaut (%)')
    axes[1].set_title('Évolution du taux de défaut mensuel')
    axes[1].legend()
    axes[1].tick_params(axis='x', rotation=45)

    # 3. Volume de dossiers
    axes[2].bar(monthly_data['DATE_MONTH'], monthly_data['nb_dossiers'],
                color='green', alpha=0.7)
    axes[2].set_ylabel('Nombre de dossiers')
    axes[2].set_xlabel('Date')
    axes[2].set_title('Volume de dossiers par mois')
    axes[2].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(FIGURES_PATH / '10_time_series_analysis.png', dpi=150)
    plt.close()
    logger.info("\n  ✓ 10_time_series_analysis.png")

    # Analyse de tendance simple
    monthly_data['month_num'] = range(len(monthly_data))
    correlation = monthly_data['month_num'].corr(monthly_data['defaults_origination'])

    if correlation > 0.3:
        trend = "croissante"
    elif correlation < -0.3:
        trend = "décroissante"
    else:
        trend = "stable"

    logger.info(f"\nAnalyse de tendance:")
    logger.info(f"  Corrélation temps/défauts: {correlation:.3f}")
    logger.info(f"  Tendance: {trend}")

    # Saisonnalité (moyenne par mois de l'année)
    monthly_data['month_of_year'] = pd.to_datetime(monthly_data['DATE_MONTH']).dt.month
    seasonality = monthly_data.groupby('month_of_year')['defaults_origination'].mean()

    fig, ax = plt.subplots(figsize=(10, 6))
    seasonality.plot(kind='bar', ax=ax, color='teal', alpha=0.8)
    ax.axhline(seasonality.mean(), color='red', linestyle='--', label='Moyenne')
    ax.set_xlabel('Mois de l\'année')
    ax.set_ylabel('Moyenne des défauts')
    ax.set_title('Saisonnalité des défauts (moyenne par mois)')
    ax.set_xticklabels(['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Jun',
                        'Jul', 'Aoû', 'Sep', 'Oct', 'Nov', 'Déc'], rotation=45)
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_PATH / '11_seasonality_analysis.png', dpi=150)
    plt.close()
    logger.info("  ✓ 11_seasonality_analysis.png")

    return monthly_data, {'correlation': correlation, 'trend': trend}


# =============================================================================
# SAUVEGARDE
# =============================================================================

def save_results(df, model, scaler, metrics, feature_importance, feature_cols, label_encoders, ts_analysis):
    """Sauvegarde tous les résultats."""
    logger.info("=" * 60)
    logger.info("SAUVEGARDE DES RÉSULTATS")
    logger.info("=" * 60)

    # Données avec features
    features_file = ROOT_DIR / 'data' / 'processed' / 'features_engineered.parquet'
    df.to_parquet(features_file, index=False)
    logger.info(f"  ✓ Features: {features_file}")

    # Modèle
    model_file = MODELS_PATH / 'logistic_regression_baseline.joblib'
    joblib.dump({
        'model': model,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'label_encoders': label_encoders
    }, model_file)
    logger.info(f"  ✓ Modèle: {model_file}")

    # Métriques
    metrics_file = REPORTS_PATH / 'model_metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump({
            'model': 'LogisticRegression',
            'metrics': {k: float(v) for k, v in metrics.items()},
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    logger.info(f"  ✓ Métriques: {metrics_file}")

    # Feature importance
    importance_file = REPORTS_PATH / 'feature_importance.csv'
    feature_importance.to_csv(importance_file, index=False)
    logger.info(f"  ✓ Feature importance: {importance_file}")

    # Rapport Phase 1
    report_file = REPORTS_PATH / 'phase1_report.md'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# Rapport Phase 1 - Scoring de Crédit\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"**Auteur**: Shafin Hamjah\n\n")

        f.write("## 1. Données\n\n")
        f.write(f"- Observations: {len(df):,}\n")
        f.write(f"- Features: {len(feature_cols)}\n")
        f.write(f"- Période: 2021-03 à 2025-12\n\n")

        f.write("## 2. Variable cible\n\n")
        f.write(f"- Non défaut (0): {(df['TARGET']==0).sum():,} ({(df['TARGET']==0).mean()*100:.2f}%)\n")
        f.write(f"- Défaut (1): {(df['TARGET']==1).sum():,} ({(df['TARGET']==1).mean()*100:.2f}%)\n\n")

        f.write("## 3. Feature Engineering\n\n")
        f.write("### Features créées:\n")
        f.write("- 11 features financières (ratios, capacité de remboursement)\n")
        f.write("- 7 features comportementales (transactions, digital)\n")
        f.write("- 7 features de stabilité (soldes, emploi, résidence)\n")
        f.write("- 6 features de risque (score custom, flags)\n")
        f.write("- 5 features d'interaction\n\n")

        f.write("## 4. Modèle Baseline (Régression Logistique)\n\n")
        f.write("### Configuration:\n")
        f.write("- Split temporel (train < 2025-01, test >= 2025-01)\n")
        f.write("- SMOTE pour rééquilibrage\n")
        f.write("- StandardScaler pour normalisation\n\n")

        f.write("### Performances sur le test:\n\n")
        f.write("| Métrique | Valeur |\n")
        f.write("|----------|--------|\n")
        f.write(f"| Accuracy | {metrics['accuracy']:.4f} |\n")
        f.write(f"| Precision | {metrics['precision']:.4f} |\n")
        f.write(f"| Recall | {metrics['recall']:.4f} |\n")
        f.write(f"| F1-Score | {metrics['f1']:.4f} |\n")
        f.write(f"| **AUC-ROC** | **{metrics['auc_roc']:.4f}** |\n\n")

        f.write("### Top 10 Features:\n\n")
        for i, row in feature_importance.tail(10).iloc[::-1].iterrows():
            f.write(f"- {row['feature']}: {row['importance']:.4f}\n")

        f.write("\n## 5. Analyse Temporelle\n\n")
        f.write(f"- Tendance: {ts_analysis['trend']}\n")
        f.write(f"- Corrélation temps/défauts: {ts_analysis['correlation']:.3f}\n\n")

        f.write("## 6. Fichiers générés\n\n")
        f.write("- `data/processed/features_engineered.parquet`\n")
        f.write("- `outputs/models/logistic_regression_baseline.joblib`\n")
        f.write("- `outputs/reports/model_metrics.json`\n")
        f.write("- `outputs/figures/08-11_*.png`\n\n")

        f.write("## 7. Prochaines étapes (Phase 2)\n\n")
        f.write("- [ ] Optimisation hyperparamètres\n")
        f.write("- [ ] Modèles avancés (Random Forest, XGBoost)\n")
        f.write("- [ ] Feature selection\n")
        f.write("- [ ] Modèle SARIMA pour séries temporelles\n")

    logger.info(f"  ✓ Rapport: {report_file}")


def main():
    """Fonction principale."""
    logger.info("=" * 60)
    logger.info("PHASE 1 - FEATURE ENGINEERING ET MODÉLISATION")
    logger.info(f"Début: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    # 1. Charger les données
    df = load_cleaned_data()

    # 2. Feature Engineering
    df = engineer_features(df)

    # 3. Préparer les données
    X_train, X_test, y_train, y_test, feature_cols, label_encoders = prepare_modeling_data(df)

    # 4. Entraîner le modèle
    model, scaler, metrics, y_test, y_pred, y_pred_proba = train_logistic_regression(
        X_train, X_test, y_train, y_test, feature_cols
    )

    # 5. Visualisations
    feature_importance = plot_model_results(y_test, y_pred, y_pred_proba, metrics, feature_cols, model)

    # 6. Analyse temporelle
    monthly_data, ts_analysis = analyze_time_series(df)

    # 7. Sauvegarder
    save_results(df, model, scaler, metrics, feature_importance, feature_cols, label_encoders, ts_analysis)

    logger.info("=" * 60)
    logger.info("PHASE 1 TERMINÉE AVEC SUCCÈS")
    logger.info(f"Fin: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    return df, model, metrics


if __name__ == "__main__":
    df, model, metrics = main()
