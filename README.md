# PRÉDICTION DU CHURN CLIENT AVEC DES MODÈLES DE MACHINE LEARNING

## 📋 Contexte du Projet

Vous êtes embauché en tant que data scientist au sein d'une entreprise de télécommunications. Votre mission consiste à développer un modèle de Machine Learning capable de prédire le **churn des clients**, c'est-à-dire le risque qu'ils résilient leur abonnement.

### Objectifs Principaux :
- Explorer les données des clients
- Sélectionner et entraîner différents modèles de régression et de classification
- Optimiser ces modèles pour obtenir les meilleures prédictions possibles
- Présenter les résultats sous forme de visualisations claires et d'une documentation détaillée

## 📊 Source des Données

**Jeu de données :** `WA_Fn-UseC_-Telco-Customer-Churn.csv`

**Lien de référence :** [IBM Community - Telco Customer Churn](https://community.ibm.com/community/user/blogs/steven-macko/2019/07/11/telco-customer-churn-1113)

## 🔍 Analyse Exploratoire des Données

### Structure du Dataset
- **7043 observations** (lignes)
- **21 variables** (colonnes)
- **Types de données :**
  - 1 valeur numérique décimale (`MonthlyCharges`)
  - 2 valeurs numériques entières (`SeniorCitizen`, `tenure`)
  - 18 valeurs textuelles/chaînes de caractères
  - Aucune valeur manquante détectée
  - Aucune valeur dupliquée

### Variables Clés Identifiées

#### Variable Cible (Target) :
- **`Churn`** : Variable catégorielle binaire définissant les catégories "Churné" et "Non-churné"

#### Variables Continues :
- **`tenure`** : Ancienneté du client (légère asymétrie à droite)
- **`MonthlyCharges`** : Charges mensuelles (asymétrie à gauche)
- **`TotalCharges`** : Charges totales (forte asymétrie à droite) - convertie en type numérique

### Distribution du Churn
- **Déséquilibre important** entre les classes :
  - Clients non-churnés : Majorité
  - Clients churnés : Minorité
- **Nécessité de rééquilibrage** pour éviter les biais de prédiction

## 🛠️ Préparation des Données

### Transformations Effectuées :
- Conversion de `TotalCharges` en type numérique
- Analyse des distributions asymétriques nécessitant potentiellement un traitement

## 📈 Analyse Descriptive

### Analyse Univariée
- **Examen individuel** de chaque variable
- **Résumés statistiques** pour comprendre les distributions
- **Identification des patterns** et caractéristiques des données

### Problème de Machine Learning Identifié
- **Type :** Classification binaire supervisée
- **Justification :** Variable cible avec seulement deux valeurs ("No", "Yes")

## 🔧 Outils et Bibliothèques

### Technologies Utilisées :
- **Python** avec Jupyter Notebook
- **Pandas** pour la manipulation des données
- **Matplotlib** et **Seaborn** pour la visualisation
- **Scikit-learn** pour le machine learning
- **SciPy** pour les tests statistiques

### Modèles de Machine Learning Importés :
- `LogisticRegression`
- `RandomForestClassifier` 
- `DecisionTreeClassifier`

## 🎯 Prochaines Étapes (Déduites du Code)

### Plan d'Action :
1. **Analyse bivariée** et multivariée
2. **Prétraitement** des variables catégorielles
3. **Feature engineering** et sélection
4. **Division** des données en ensembles d'entraînement/test
5. **Entraînement** des modèles de classification
6. **Évaluation** et optimisation des performances
7. **Gestion du déséquilibre** des classes

## 📊 Métriques d'Évaluation Prévues

D'après les imports, les métriques suivantes seront utilisées :
- `accuracy_score`
- `classification_report`
- `roc_auc_score`
- `roc_curve`

## 🚀 Installation et Exécution

```bash
# Cloner le repository
git clone [repository-url]

# Naviguer vers le dossier du projet
cd prediction_du_churn_client

# Installer les dépendances
pip install -r requirements.txt

# Structure des dossiers pour lancer le notebook Jupyter
jupyter notebook prediction_du_churn_client.ipynb
📁 Structure du Projet
PREDICTION_DU_CHURN_CLIENT_AVEC_MODELE_DE_MACHINE_LEARNING
env
etl/
└── __init__.py
└── extract.py
└── load.py
└── transform.py
infrastructure/
└── data_warehouse/
    └── BI_data/
        └── WA_Fn-UseC_-Telco-Customer-Churn.csv
notebook/
└── prediction_du_churn_client.ipynb                Il est ici le notebook
visualisations/
└── app_viz.py
README.md
requirements.txt
Ce projet vise à fournir une solution robuste de prédiction du churn client, essentielle pour la rétention client dans le secteur des télécommunications.

text

## Méthode 3 : Si vous utilisez GitHub
Si vous avez un repository GitHub, vous pouvez :
1. Créer le fichier directement sur GitHub
2. Utiliser l'interface web pour créer `README.md`
3. Coller le contenu

Le fichier est maintenant prêt à être utilisé dans votre projet !
