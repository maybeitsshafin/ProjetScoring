# Scoring de Crédit et Prévision de Défauts

**Projet Tutoré - Licence Professionnelle Data Mining**
Université Gustave Eiffel | 2025-2026

## Description

Ce projet vise à développer un système de scoring de crédit pour prédire la probabilité de défaut des clients, ainsi qu'un modèle de prévision temporelle des volumes de défauts.

## Équipe

- **Shafin Hamjah** - Chef de projet / Data Engineer
- **Ismaël Naït Daoud** - Chef de projet / Data Engineer
- **Manon Trujillo** - Data Scientist
- **Virginie Lawrence Raj** - Data Analyst / Data Engineer
- **Ange Ngah** - Data Scientist

## Installation

```bash
# Cloner le repository
git clone <url-du-repo>
cd projet-scoring-credit

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Installer les dépendances
pip install -r requirements.txt
```

## Structure du projet

```
projet-scoring-credit/
├── data/
│   ├── raw/          # Données brutes
│   ├── processed/    # Données nettoyées
│   └── external/     # Données externes
├── notebooks/        # Jupyter notebooks
├── src/              # Code source
│   ├── preprocessing/
│   ├── features/
│   ├── models/
│   └── utils/
├── outputs/          # Résultats
│   ├── models/
│   ├── figures/
│   ├── reports/
│   └── predictions/
├── tests/            # Tests unitaires
└── docs/             # Documentation
```

## Utilisation

Voir le fichier `CLAUDE.md` pour les instructions détaillées.

## Encadrant

M. Bosco

## Licence

Projet académique - Université Gustave Eiffel
