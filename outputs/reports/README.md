# Structure des Rapports

Organisation des rapports générés par le projet de scoring de crédit.

## Arborescence

```
reports/
├── 01_exploration/          # Analyse exploratoire des données
│   ├── rapport_exploration.json      # Statistiques générales du dataset
│   ├── rapport_outliers.json         # Analyse des valeurs aberrantes
│   └── rapport_qualite_donnees.csv   # Rapport qualité (types, nulls, stats)
│
├── 02_preprocessing/        # Prétraitement des données
│   └── resume_preprocessing.md       # Résumé des étapes de nettoyage
│
├── 03_modeling/             # Modélisation et résultats
│   ├── rapport_phase1_complet.md     # Rapport complet Phase 1
│   ├── metriques_modele_baseline.json # Métriques du modèle (AUC, F1, etc.)
│   └── importance_variables.csv      # Classement des features par importance
│
└── logs/                    # Fichiers de logs d'exécution
    ├── 01_preprocessing.log          # Log du script de preprocessing
    └── 02_modeling.log               # Log du script de modélisation
```

## Description des fichiers

### 01_exploration/

| Fichier | Description |
|---------|-------------|
| `rapport_exploration.json` | Dimensions, types, distribution TARGET |
| `rapport_outliers.json` | Variables avec outliers > 1% (méthode IQR) |
| `rapport_qualite_donnees.csv` | Qualité par colonne (nulls, unique, min/max) |

### 02_preprocessing/

| Fichier | Description |
|---------|-------------|
| `resume_preprocessing.md` | Actions de nettoyage effectuées |

### 03_modeling/

| Fichier | Description |
|---------|-------------|
| `rapport_phase1_complet.md` | Rapport final Phase 1 (features, modèle, résultats) |
| `metriques_modele_baseline.json` | AUC-ROC, Precision, Recall, F1, Accuracy |
| `importance_variables.csv` | Top features avec coefficients |

---

*Dernière mise à jour : Janvier 2026*
