# Playground ML

Bienvenue dans **Playground ML**, une application web interactive conçue pour vous permettre d'explorer le machine learning de manière intuitive. Que vous soyez débutant ou expert, notre application vous guide à travers chaque étape, de la préparation des données à l'évaluation des modèles.

## Objectif du projet

Ce projet a été réalisé dans le cadre du **Projet 2 Diginamic Lyon** pour le **POEC Data Analyst**. L'objectif est de construire un **playground ML** interactif afin de mettre en pratique nos 5 jours de cours sur le Machine Learning.

Le langage de programmation utilisé est **Python**.

- **Date du projet** : du 04/09/2024 au 10/09/2024

## Contributeurs

- **Melissa KUNEGEL**
- **Lucas FANGET**
- **Grégoire DELCROIX**

**Intervenant** :
- Bastien ANGELOZ, data scientist

---

## Fonctionnalités principales

### Import de données
- Importez vos fichiers CSV pour commencer l'analyse.
- Chargement de jeux de données d'exemple (vin.csv pour la Classification, diabete.csv pour la Régression).

### Prétraitement des données
- Nettoyage des données avec options d'imputation des valeurs manquantes.
- Encodage des variables catégorielles (OneHotEncoding, Ordinal Encoding).
- Normalisation et standardisation des données.

### Exploration des données
- Visualisez les relations entre vos variables (heatmaps, pairplots, boxplots).
- Sélectionnez vos variables explicatives et cible via une interface intuitive.

### Modélisation
- Appliquez des modèles de classification (régression logistique, KNN, Random Forest).
- Testez différents modèles avec LazyPredict ou LazyRegressor et comparez les performances.
- Accédez également aux algorithmes de régression.
- Acculturation au DeepLearning avec l'API Roboflow pour la détection des ongles à partir d'une image

### Téléchargement de rapports
- Génération de rapports récapitulatifs avec les métriques des modèles.
- Téléchargez des fichiers CSV avec les résultats de vos analyses.

## Installation

### Pré-requis
- Python 3.10.9
- Streamlit 1.38.0
- Pandas, Scikit-learn, Numpy, LazyPredict

### Étapes d'installation

1. Clonez le projet :
    ```bash
    git clone https://github.com/mkunegel/ProjetML.git
    cd ProjetML
    ```

2. Créez un environnement virtuel (MacOS, Windows):
   ```bash
    python3 -m venv .venv
    source .venv/bin/activatec
    ```
   
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. Installez les dépendances :
    ```bash
    pip install -r requirements.txt
    ```

### Lancement de l'application

1. Exécutez le fichier principal avec Streamlit :
    ```bash
    streamlit run app.py
    ```

2. Utilisez l'interface web pour :
    - **Importer des données**.
    - **Nettoyer et explorer vos données**.
    - **Appliquer des modèles de machine learning**.

## Structure du projet

```bash
ProjetML/
│
├── app.py                       # Fichier principal pour lancer l'application
├── requirements.txt             # Dépendances du projet
├── sections/
│   ├── classification/          # Modèles et analyses de classification
│   ├── dataPreprocessing/        # Prétraitement des données
│   ├── dataExplore/              # Visualisations et statistiques descriptives
│   ├── nailsdetection/           # Section pour la détection des ongles via API Roboflow
│   └── regression/               # Modèles de régression
├── data/                        # Contient les datasets d'exemple
└── README.md                    # Ce fichier !
