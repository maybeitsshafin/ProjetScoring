# Structure des Figures

Visualisations générées par le projet de scoring de crédit.

## Arborescence

```
figures/
├── 01_exploration/              # Analyse exploratoire
│   ├── distribution_target.png          # Distribution de la variable cible
│   ├── distribution_age.png             # Distribution de l'âge par TARGET
│   ├── analyse_temporelle.png           # Volume et taux de défaut par mois
│   ├── valeurs_manquantes.png           # Graphique des valeurs manquantes
│   ├── distributions_financieres.png    # Distributions LOAN, INCOME, DTI, SCORE
│   ├── taux_defaut_par_categorie.png    # Taux de défaut par variable catégorielle
│   └── matrice_correlation.png          # Matrice de corrélation des variables clés
│
└── 02_modeling/                 # Modélisation
    ├── performance_modele.png           # ROC, Confusion Matrix, Precision-Recall
    ├── importance_features.png          # Top 20 features (coefficients)
    ├── serie_temporelle.png             # Évolution des défauts dans le temps
    └── saisonnalite.png                 # Saisonnalité mensuelle des défauts
```

## Aperçu

### 01_exploration/

| Figure | Description |
|--------|-------------|
| `distribution_target.png` | Classes déséquilibrées (81% vs 19%) |
| `distribution_age.png` | Comparaison des âges défaut/non-défaut |
| `analyse_temporelle.png` | Tendances volumes et taux de défaut |
| `valeurs_manquantes.png` | REGION : 8.5% de valeurs manquantes |
| `distributions_financieres.png` | Histogrammes des variables financières clés |
| `taux_defaut_par_categorie.png` | Taux de défaut par SEX, MARITAL_STATUS, etc. |
| `matrice_correlation.png` | Corrélations entre variables numériques |

### 02_modeling/

| Figure | Description |
|--------|-------------|
| `performance_modele.png` | AUC-ROC = 0.71, matrice de confusion |
| `importance_features.png` | DTI_RATIO et INSTALLMENT_TO_INCOME en tête |
| `serie_temporelle.png` | Tendance croissante des défauts |
| `saisonnalite.png` | Pics de défauts en fin d'année |

---

*Dernière mise à jour : Janvier 2026*
