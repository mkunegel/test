import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Fonction principale pour l'apercu des données
def show_apercu(test_size, random_state):
    st.subheader("Aperçu des Données")

    # Chargement des données
    df = st.session_state['df']  # on charge les données via le state passé depuis app.py
    st.write(df)

    # Séparation des données et de la target
    X = df.drop(columns=['target'])
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Description des colonnes et affichage
    descriptions = {
        'Age': 'Âge du patient',
        'Sexe': 'Sexe du patient',
        'BMI': 'Indice de masse corporelle (IMC, Body Mass Index), qui est une mesure du rapport entre le poids et la taille d\'une personne. C\'est un indicateur de la corpulence.',
        'BP': 'Pression artérielle moyenne (Mean Arterial Pressure), qui est une mesure de la pression sanguine moyenne dans les artères au cours d\'un cycle cardiaque complet.',
        's1': 'Taux de cholestérol total (TC, Total Cholesterol)',
        's2': 'Taux de lipoprotéines de basse densité (LDL, Low-Density Lipoprotein Cholesterol)',
        's3': 'Taux de lipoprotéines de haute densité (HDL, High-Density Lipoprotein Cholesterol)',
        's4': 'Rapport cholestérol total / cholestérol HDL (TCH, Total Cholesterol / HDL ratio)',
        's5': 'Taux de triglycérides logarithmiques (LTG, Logarithm of Triglycerides)',
        's6': 'Taux de glucose sanguin (GLU, Blood Glucose)'
    }
    st.subheader("Descriptions des Colonnes")
    for column, description in descriptions.items():
        st.write(f"**{column}:** {description}")

    # Affichage des statistiques descriptives basiques
    st.subheader("Statistiques Descriptives")
    st.write(X.describe())

    # Affichage de la matrice de corrélation avec un popover
    with st.popover("Afficher la Matrice de Corrélation"):  # Utilisation de st.expander pour contenir le graphique
        corr_matrix = X.corr()
        fig, ax = plt.subplots(figsize=(10, 6))  # Ajuste la taille du graphique
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
        ax.set_title('Matrice de Corrélations', fontsize=14)
        st.pyplot(fig)

    # Explication rapide pour les corrélations
    st.write(
        """
        **Corrélation forte entre S1 (cholestérol total) et S2 (cholestérol LDL) :**

        Une forte corrélation entre **S1** (cholestérol total) et **S2** (cholestérol LDL) est cohérente sur le plan médical. Le cholestérol total est la somme du cholestérol HDL, du cholestérol LDL, et d'une fraction des triglycérides. Le cholestérol LDL, souvent appelé "mauvais cholestérol", constitue une partie importante du cholestérol total. Par conséquent, un niveau élevé de LDL est généralement associé à un niveau élevé de cholestérol total, ce qui explique la forte corrélation observée.
        """
    )

    st.write(
        """
        **Corrélation faible entre S1 (cholestérol total) et S3 (cholestérol HDL) :**

        Une faible corrélation entre **S1** (cholestérol total) et **S3** (cholestérol HDL) est également cohérente. Le cholestérol HDL, souvent appelé "bon cholestérol", a un effet protecteur contre les maladies cardiovasculaires et peut varier indépendamment des niveaux de cholestérol total. Le cholestérol total reflète une somme des niveaux de différentes fractions de cholestérol, y compris LDL et HDL, et n'est pas directement proportionnel aux niveaux de HDL. Ainsi, le HDL peut fluctuer sans que le cholestérol total change de manière proportionnelle.
        """
    )

    return X_train, X_test, y_train, y_test