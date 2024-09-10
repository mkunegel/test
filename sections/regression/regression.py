import streamlit as st
import pandas as pd

from sections.regression.apercu import show_apercu
from sections.regression.modeles import show_modeles
from sections.regression.visuels import show_visuels
from sections.regression.comparaison import show_comparaison

# Fonction principale pour le Playground de Régression
def regression_page():

    st.caption("Bienvenue dans le Playground de Régression")


    if 'df' not in st.session_state:
        # Charger le fichier CSV dans df
        st.session_state['df'] = pd.read_csv("./data/diabete.csv", index_col=0)
    df = st.session_state['df']  # Récupérer les données depuis st.session_state

    # Initialiser les clés X et y dans st.session_state si elles ne sont pas encore définies
    if 'X' not in st.session_state:
        st.session_state['X'] = None
    if 'y' not in st.session_state:
        st.session_state['y'] = None
    # Jauge de test_size dans la barre latérale
    test_size = st.sidebar.slider("Proportion du Test (en %)", min_value=5, max_value=50, value=20, step=1,
                                  help="Choisissez une valeur comprise entre 5 et 50") / 100

    # Taille totale de la population basée sur le DataFrame sélectionné
    total_population = df.shape[0]

    # Calcul des populations pour le train et le test
    test_population = int(total_population * test_size)
    train_population = total_population - test_population

    # Encadrer les résultats avec moins d'espacement entre les lignes
    st.sidebar.markdown(f"""
            <div style="border: 1px solid #aaa; padding: 10px; border-radius: 5px;">
                <p style="margin: 0;text-align: center;">Population test : {test_population}</p>
                <p style="margin: 0;text-align: center;">Population train : {train_population}</p>
                <p style="margin: 0; text-align: center;">Taille totale de la population : {total_population}</p>
            </div>
            """, unsafe_allow_html=True)

    # Jauge de random_state dans la barre latérales
    # Utilisation d'un bouton pour verrouiller/déverrouiller random_state
    if "random_state_locked" not in st.session_state:
        st.session_state.random_state_locked = True  # Par défaut, random_state est verrouillé à 42

    # Bouton pour déverrouiller le random_state
    unlock_button = st.sidebar.checkbox("Déverrouiller Random State", value=False)

    # Si le bouton est activé, permettre à l'utilisateur de saisir un random_state personnalisé
    if unlock_button:
        random_state = st.sidebar.number_input("Random State", min_value=0, max_value=100,
                                               value=st.session_state.get("custom_random_state", 42),
                                               help="Saisissez un nombre entre 0 et 100")
        st.session_state.custom_random_state = random_state  # Enregistrer la valeur personnalisée
    else:
        random_state = 42  # Par défaut, random_state est verrouillé à 42
        st.sidebar.write(f"Random State actuel : {random_state}")

    # Vérification de la valeur saisie
    if unlock_button:
        if random_state < 0 or random_state > 100:
            st.sidebar.error("Erreur : Veuillez saisir un nombre entre 0 et 100.")
        else:
            st.sidebar.write(f"Random State sélectionné : {random_state}")

    # Onglets principaux : Aperçu des données, Modèles, Visuels, Comparaison des modèles
    apercu_tab, modeles_tab, visuels_tab,comparaison_tab = st.tabs(["Aperçu des Données", "Modèles", "Visuels","Comparaison des modèles"])

    # Appel des fonctions pour chaque onglet
    with apercu_tab:
        X_train, X_test, y_train, y_test = show_apercu(test_size, random_state)

    with modeles_tab:
        show_modeles(X_train, X_test, y_train, y_test, random_state)

    with visuels_tab:
        show_visuels(X_train, X_test, y_train, y_test)

    with comparaison_tab:
        show_comparaison(X_train, X_test, y_train, y_test)

  
