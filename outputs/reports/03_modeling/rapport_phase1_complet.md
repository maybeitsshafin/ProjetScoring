# Rapport Phase 1 - Scoring de Crédit

**Date**: 2026-01-23 16:06
**Auteur**: Shafin Hamjah

## 1. Données

- Observations: 130,000
- Features: 99
- Période: 2021-03 à 2025-12

## 2. Variable cible

- Non défaut (0): 105,820 (81.40%)
- Défaut (1): 24,180 (18.60%)

## 3. Feature Engineering

### Features créées:
- 11 features financières (ratios, capacité de remboursement)
- 7 features comportementales (transactions, digital)
- 7 features de stabilité (soldes, emploi, résidence)
- 6 features de risque (score custom, flags)
- 5 features d'interaction

## 4. Modèle Baseline (Régression Logistique)

### Configuration:
- Split temporel (train < 2025-01, test >= 2025-01)
- SMOTE pour rééquilibrage
- StandardScaler pour normalisation

### Performances sur le test:

| Métrique | Valeur |
|----------|--------|
| Accuracy | 0.6989 |
| Precision | 0.3215 |
| Recall | 0.5797 |
| F1-Score | 0.4136 |
| **AUC-ROC** | **0.7105** |

### Top 10 Features:

- INSTALLMENT_TO_INCOME: 0.4272
- DTI_RATIO: 0.4272
- VAR_BALANCE_3M: 0.2198
- CREDIT_SCORE_NORM: 0.1724
- CREDIT_SCORE: 0.1724
- CHECKING_STABILITY: 0.1286
- HAS_PREVIOUS_DEFAULT: 0.1165
- LOAN_AMOUNT: 0.1100
- INSURANCE_TAKEN: 0.1082
- PHONE_VERIFIED: 0.0890

## 5. Analyse Temporelle

- Tendance: croissante
- Corrélation temps/défauts: 0.583

## 6. Fichiers générés

- `data/processed/features_engineered.parquet`
- `outputs/models/logistic_regression_baseline.joblib`
- `outputs/reports/model_metrics.json`
- `outputs/figures/08-11_*.png`

## 7. Prochaines étapes (Phase 2)

- [ ] Optimisation hyperparamètres
- [ ] Modèles avancés (Random Forest, XGBoost)
- [ ] Feature selection
- [ ] Modèle SARIMA pour séries temporelles
