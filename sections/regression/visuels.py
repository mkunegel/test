import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from lazypredict.Supervised import LazyRegressor

# Fonction principale pour la visualisation
def show_visuels(X_train, X_test, y_train, y_test):
    visuels_subtabs = st.tabs(["Matrice de Corrélation"])

    with visuels_subtabs[0]:
        show_correlation_matrix()

# Fonction pour la matrice de corrélation
def show_correlation_matrix():
    st.subheader("Matrice de Corrélation")
    df = st.session_state['df']
    corr_matrix = df.corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
    st.pyplot(fig)
