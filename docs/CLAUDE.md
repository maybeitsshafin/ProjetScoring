# CLAUDE.md - Agent Data Analyse | Projet Tutoré LP Data Mining 2025-2026

## 🎯 Contexte du Projet

**Projet** : Scoring de Crédit et Prévision de Défauts  
**Formation** : Licence Professionnelle Data Mining - Université Gustave Eiffel  
**Encadrant** : M. Bosco  
**Période** : Novembre 2025 - Juin 2026

### Équipe

| Membre | Rôle | Responsabilités |
|--------|------|-----------------|
| Shafin Hamjah | Chef de projet / Data Engineer | Coordination, préparation données, nettoyage, validation, documentation, code |
| Ismaël Naït Daoud | Chef de projet / Data Engineer | Coordination, préparation données, nettoyage, validation, documentation, code |
| Manon Trujillo | Data Scientist | Modèle de scoring, régression logistique, communication |
| Virginie Lawrence Raj | Data Analyst / Data Engineer | Feature engineering, modèles avancés (arbres, ensembles), tuning hyperparamètres |
| Ange Ngah | Data Scientist | Modèle SARIMA, prévisions temporelles, visualisations, analyse stationnarité |

---

## 📊 Description des Données

### Vue d'ensemble
- **Volume** : ~130 000 lignes
- **Variables** : ~70 colonnes
- **Source** : Dossiers clients bancaires (octroi de crédits)

### Variable Cible
```
TARGET : variable binaire
  - 0 = Client non en défaut
  - 1 = Client en défaut au moment de l'octroi
```

⚠️ **ATTENTION** : Classes déséquilibrées - prévoir techniques de rééquilibrage (SMOTE, class_weight, etc.)

### Catégories de Variables

#### Informations démographiques
- `age` : Âge du client
- `sexe` : Genre
- `statut_marital` : Situation familiale
- `nb_enfants` : Nombre d'enfants
- `niveau_education` : Niveau d'études
- `region` : Région géographique

#### Informations professionnelles
- `type_emploi` : Catégorie d'emploi
- `anciennete` : Ancienneté dans l'emploi
- `statut_travailleur_etranger` : Indicateur travailleur étranger

#### Informations financières
- `revenus_mensuels` : Revenus mensuels
- `depenses` : Dépenses mensuelles
- `montant_pret` : Montant du prêt demandé
- `taux_interet` : Taux d'intérêt du prêt
- `mensualites` : Montant des mensualités
- `DTI` : Ratio d'endettement (Debt-to-Income)

#### Historique bancaire (3 derniers mois)
- `solde_epargne_M3`, `solde_epargne_M2`, `solde_epargne_M1` : Soldes compte épargne
- `solde_courant_M3`, `solde_courant_M2`, `solde_courant_M1` : Soldes compte courant
- `variance_soldes` : Variance des soldes
- `retards_paiement` : Historique de retards

#### Comportement de crédit
- `score_credit_externe` : Score de crédit externe
- `nb_prets_ouverts` : Nombre de prêts en cours
- `taux_utilisation_credit` : Taux d'utilisation du crédit disponible
- `historique_defauts` : Historique de défauts passés

#### Données comportementales
- Transactions : `transactions_POS`, `transactions_ATM`, `transactions_online`
- Connexions : `connexions_mobile`, `connexions_web`
- `visites_agence` : Fréquence des visites en agence

#### Variables temporelles
- `DATE_MONTH` : Mois d'origination du crédit (clé pour split temporel)
- `DEFAULTS_ORIGINATION` : Nombre total de défauts par mois (pour série temporelle)

---

## 🔬 Problématiques et Objectifs

### Axe 1 : Modèle de Scoring (Prédiction de la Probabilité de Défaut - PD)

**Objectif** : Classification binaire pour estimer P(TARGET=1) à l'origination

**Enjeux métier** :
- Prioriser les dossiers à risque avant décision
- Calibrer les politiques d'acceptation et limites de crédit
- Optimiser la rentabilité (équilibre risque/volume)

**Modèles à implémenter** (par ordre de priorité) :
1. Régression Logistique (baseline, interprétable)
2. Arbre de décision (interprétabilité)
3. Random Forest
4. XGBoost/LightGBM

**Métriques d'évaluation** :
- AUC-ROC (métrique principale)
- Précision / Rappel / F1-score
- Courbe de Lift
- KS Statistic
- Matrice de confusion

### Axe 2 : Modèle de Série Temporelle (Prévision des Volumes de Défauts)

**Objectif** : Prévoir `DEFAULTS_ORIGINATION` sur les 3 prochains mois

**Approche méthodologique** :
1. Tests de stationnarité (ADF, KPSS)
2. Analyse ACF/PACF (corrélogrammes)
3. Différenciation si nécessaire
4. Modélisation ARIMA/SARIMA(p,d,q)(P,D,Q,12)
5. Prévision avec intervalles de confiance

**Livrables attendus** :
- Visualisation historique + prévisions + IC
- Métriques : RMSE, MAE, MAPE

---

## 🛠️ Stack Technique

### Langages
- **Python** (principal)
- R (optionnel pour certaines analyses)

### Bibliothèques Python

```python
# Data manipulation
pandas>=2.0
numpy>=1.24
polars  # optionnel, pour performances

# Visualisation
matplotlib>=3.7
seaborn>=0.12
plotly>=5.15

# Machine Learning
scikit-learn>=1.3
imbalanced-learn>=0.11  # SMOTE, etc.
xgboost>=2.0
lightgbm>=4.0

# Séries temporelles
statsmodels>=0.14  # ARIMA, SARIMA, tests ADF
pmdarima>=2.0  # auto_arima

# Utilitaires
pandera>=0.17  # validation de données
shap>=0.42  # interprétabilité
joblib>=1.3  # sauvegarde modèles
tqdm>=4.65  # barres de progression
```

### Outils
- **IDE** : VS Code avec Claude Code
- **Versioning** : Git (commits conventionnels)
- **Notebooks** : Jupyter pour exploration
- **Documentation** : Markdown, docstrings Google format

---

## 📁 Structure du Projet

```
projet-scoring-credit/
│
├── CLAUDE.md                    # Ce fichier - instructions agent
├── README.md                    # Documentation projet
├── requirements.txt             # Dépendances Python
├── .gitignore
│
├── data/
│   ├── raw/                     # Données brutes (ne pas modifier)
│   │   └── credit_data.csv
│   ├── processed/               # Données nettoyées
│   │   ├── train.parquet
│   │   ├── test.parquet
│   │   └── features_engineered.parquet
│   └── external/                # Données externes si besoin
│
├── notebooks/
│   ├── 01_exploration.ipynb     # EDA initial
│   ├── 02_preprocessing.ipynb   # Nettoyage et imputation
│   ├── 03_feature_engineering.ipynb
│   ├── 04_modeling_scoring.ipynb
│   ├── 05_modeling_timeseries.ipynb
│   └── 06_evaluation_finale.ipynb
│
├── src/
│   ├── __init__.py
│   ├── config.py                # Paramètres globaux, chemins
│   │
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── cleaning.py          # Nettoyage données
│   │   ├── imputation.py        # Gestion valeurs manquantes
│   │   └── validation.py        # Contrôles qualité
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── engineering.py       # Création de features
│   │   ├── selection.py         # Sélection de variables
│   │   └── encoders.py          # Encodage catégorielles
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── scoring.py           # Modèles de classification
│   │   ├── timeseries.py        # Modèles SARIMA
│   │   └── evaluation.py        # Métriques et évaluation
│   │
│   └── utils/
│       ├── __init__.py
│       ├── io.py                # Lecture/écriture fichiers
│       ├── plots.py             # Fonctions de visualisation
│       └── logger.py            # Configuration logging
│
├── outputs/
│   ├── models/                  # Modèles sauvegardés (.joblib)
│   ├── figures/                 # Graphiques exportés
│   ├── reports/                 # Rapports générés
│   └── predictions/             # Prédictions finales
│
├── tests/                       # Tests unitaires
│   ├── test_preprocessing.py
│   ├── test_features.py
│   └── test_models.py
│
└── docs/
    ├── rapport_intermediaire.md  # Livrable Mars 2026
    └── rapport_final.md          # Livrable Juin 2026
```

---

## ⚙️ Conventions et Règles

### Règles Métier Critiques

1. **Séparation temporelle obligatoire** : Utiliser `DATE_MONTH` pour séparer train/test (pas de random split)
2. **Pas de data leakage** : Ne jamais utiliser de variables post-origination pour le scoring
3. **Interprétabilité** : Privilégier des modèles explicables (régression logistique en baseline)
4. **Documentation** : Chaque feature créée doit être documentée (nom, formule, justification)

### Conventions de Code

```python
# Type hints obligatoires
def calculate_dti(income: float, debt: float) -> float:
    """
    Calcule le ratio d'endettement (Debt-to-Income).
    
    Args:
        income: Revenu mensuel du client
        debt: Total des dettes mensuelles
        
    Returns:
        Ratio DTI en pourcentage
        
    Raises:
        ValueError: Si income <= 0
    """
    if income <= 0:
        raise ValueError("Le revenu doit être positif")
    return (debt / income) * 100
```

### Logging (pas de print)

```python
import logging

logger = logging.getLogger(__name__)
logger.info("Chargement des données...")
logger.warning(f"Valeurs manquantes détectées: {missing_count}")
logger.error("Échec du chargement du fichier")
```

### Commits Git (Conventional Commits)

```
feat: ajout du modèle de régression logistique
fix: correction du calcul DTI pour revenus nuls
docs: mise à jour README avec instructions installation
refactor: restructuration module preprocessing
test: ajout tests unitaires feature engineering
```

---

## 🚨 Risques Identifiés et Solutions

| Risque | Impact | Solution |
|--------|--------|----------|
| Classes déséquilibrées (TARGET) | Modèle biaisé vers classe majoritaire | SMOTE, class_weight, threshold tuning |
| Nombreuses valeurs manquantes | Perte d'information, biais | Imputation multiple, analyse de sensibilité, indicateurs de missingness |
| Non-stationnarité série temporelle | Prévisions incorrectes | Tests ADF/KPSS, différenciation, transformations |
| Surapprentissage | Mauvaise généralisation | Validation croisée temporelle (TimeSeriesSplit), régularisation, early stopping |
| Data leakage | Métriques faussement optimistes | Revue rigoureuse des features, split temporel strict |

---

## 📅 Planning et Livrables

### Phase 1 : Novembre - Décembre 2025
- [ ] Exploration approfondie des données
- [ ] Analyse de qualité (valeurs manquantes, outliers, distributions)
- [ ] Prétraitement complet
- [ ] Feature engineering initial
- [ ] Première modélisation scoring (baseline régression logistique)
- [ ] Analyse temporelle préliminaire

### Phase 2 : Janvier - Mars 2026 (Livrable intermédiaire)
- [ ] Optimisation modèles scoring (tuning, validation croisée)
- [ ] Feature engineering avancé
- [ ] Mise en place modèle SARIMA
- [ ] Évaluation comparative des modèles
- [ ] Tests de robustesse
- [ ] **Livrable intermédiaire (Mars 2026)**

### Phase 3 : Avril - Juin 2026 (Livrable final)
- [ ] Finalisation des modèles
- [ ] Prévisions 3 mois avec IC
- [ ] Rédaction rapport final
- [ ] Dashboards et visualisations
- [ ] Documentation code
- [ ] **Soutenance finale (Juin 2026)**

---

## 🤖 Instructions pour l'Agent Claude Code

### Comportement général
- Toujours vérifier la structure des données avant manipulation (`df.info()`, `df.head()`)
- Privilégier pandas pour la manipulation, scikit-learn pour le ML
- Générer des logs informatifs, pas de print()
- Sauvegarder les résultats intermédiaires en .parquet (plus efficace que CSV)
- Créer des visualisations claires avec titres et labels

### Workflow d'analyse standard
1. **Chargement** : Vérifier types, dimensions, premières lignes
2. **Qualité** : `df.isnull().sum()`, `df.describe()`, détection outliers
3. **Exploration** : Distribution TARGET, corrélations, visualisations
4. **Préparation** : Imputation, encoding, scaling
5. **Split** : Séparation temporelle sur `DATE_MONTH`
6. **Modélisation** : Baseline puis modèles avancés
7. **Évaluation** : Métriques multiples, courbes, matrices
8. **Documentation** : Rapport markdown avec conclusions

### Commandes fréquentes
```bash
# Lancer l'exploration
claude "Charge les données et génère un rapport EDA complet"

# Feature engineering
claude "Crée des features financières : ratios, indicateurs de stabilité"

# Modélisation
claude "Entraîne une régression logistique avec validation croisée temporelle"

# Évaluation
claude "Compare les performances des modèles et génère les courbes ROC"
```

### Chemins importants
- Données brutes : `data/raw/`
- Données traitées : `data/processed/`
- Modèles : `outputs/models/`
- Figures : `outputs/figures/`
- Rapports : `outputs/reports/`

---

## 📞 Contacts et Ressources

- **Encadrant** : M. Bosco
- **Repository Git** : [À compléter]
- **Documentation scikit-learn** : https://scikit-learn.org/stable/
- **Documentation statsmodels** : https://www.statsmodels.org/stable/

---

*Dernière mise à jour : 28/11/2025*
