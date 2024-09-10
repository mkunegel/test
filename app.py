import streamlit as st
import numpy as np
import random
from sections.dataImport.dataImport import sourceData_page
from sections.dataPreprocessing.dataPreprocessing import nettoyageData_page
from sections.classification.classification import classification_page
from sections.dataExplore.dataExplore import apercuData_page
from sections.nailsdetection.nails import nail_page
from sections.regression.regression import regression_page

st.set_page_config(
    page_title="Playground ML",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Fonction pour lire le fichier README.md
def read_readme():
    with open("README.md", "r", encoding="utf-8") as file:
        readme_text = file.read()
    return readme_text

# Ajouter un bouton de reset dans la sidebar
if st.sidebar.button("🔄 Reset APP avant Playground"):
    # Réinitialiser les variables souhaitées dans st.session_state
    for key in st.session_state.keys():
        del st.session_state[key]
    st.sidebar.success("Les variables ont été réinitialisées.")

# Logo diginamic
st.image("banniereapp.jpg", use_column_width=True)

# Barre horizontale en haut
st.markdown("""
    <style>
    .top-bar {
        background-color: #F0F2F6;
        padding: 10px;
        font-size: 18px;
        font-weight: bold;
        text-align: right;
    }
    .top-bar a {
        margin: 0 30px;
        text-decoration: none;
        color: #F90100;
    }
    .top-bar a:hover {
        color: #0056b3;
    }
    </style>
    <div class="top-bar">
        <a href="https://github.com/mkunegel/ProjetML" target="_blank">🔗 Lien GitHub projet ML</a>
    </div>
    """, unsafe_allow_html=True)

st.markdown("# Projet Machine Learning 🎈")

# Barre de navigation principale
st.sidebar.title("Barre de Navigation")
page = st.sidebar.selectbox(
    "Choisissez une page",
    ["Accueil", "README", "Source de données", "Nettoyage des données", "Aperçu du dataset", "Playground"]
)

# Gestion des pages
if page == "Accueil":
    st.write("## Bienvenue 👋🏻 sur l'application Machine Learning Playground ! 🎉")
    st.write("""
    Nous sommes ravis de vous accueillir sur notre plateforme interactive dédiée au Machine Learning. 
    Explorez nos différentes sections via la barre latérale, découvrez les modèles, et apprenez tout sur la science des données de manière ludique et intuitive.

    Nous vous souhaitons une excellente navigation à travers notre application. Profitez bien de votre expérience, et n'hésitez pas à revenir nous voir régulièrement pour des nouveautés !

    **Bonne navigation !**
    """)
    st.write("### Melissa, Lucas & Grégoire 👨‍💻👩‍💻")

    st.markdown("---")

    # Ajouter le chatbot dans un expander
    with st.expander("💬 Chatbot - Posez vos questions ici"):
        # Initialiser les messages si ce n'est pas déjà fait
        if 'messages' not in st.session_state:
            st.session_state['messages'] = []

        if 'first_interaction' not in st.session_state:
            st.session_state['first_interaction'] = True  # Pour gérer la première interaction

        # Fonction pour afficher les messages
        def display_messages():
            for message in st.session_state['messages']:
                with st.chat_message(message["role"]):
                    st.write(message["content"])

        # Réponses drôles pour la première interaction
        def first_interaction_response():
            responses = [
                "Salut ! Ravi de te rencontrer ! Je suis ton assistant Machine Learning, et je suis là pour te guider avec un peu d'humour 🤓. Comment puis-je t'aider aujourd'hui ?",
                "Hey, t'es là ! J'espère que t'es prêt à explorer l'univers des données avec moi. On va bien s'amuser 🚀!",
                "Bienvenue ! Je suis le bot le plus cool de cette plateforme 😎. Pose-moi tes questions, et on va déchirer ensemble !"
            ]
            return random.choice(responses)

        # Fonction pour générer une réponse basée sur la question de l'utilisateur
        def faq_response(user_input):
            faq = {
                "Comment importer des données ?": "Facile ! Va dans la section 'Source de données' et clique sur 'Importer'. Si tu n'y arrives pas, c'est sûrement parce que tu n'as pas encore fait de café ☕.",
                "Comment nettoyer les données ?": "Ah, nettoyer les données, c'est comme passer l'aspirateur dans une chambre d'ado ! Utilise la section 'Nettoyage des données'.",
                "Quels algorithmes de classification sont disponibles ?": "Tu as de la chance, nous avons plein de choix : Régression logistique, Forêts aléatoires, et bien plus ! 🧠",
                "Comment explorer les données ?": "Si tu veux explorer les données, fonce dans la section 'Aperçu du dataset'. Tu y trouveras de belles visualisations 📊.",
                "Qu'est-ce que le Playground ?": "Le Playground, c'est comme un parc d'attractions 🎢 pour les données. Amuse-toi à tester différents modèles de machine learning.",
                "Comment réinitialiser l'application ?": "Besoin de tout recommencer ? Clique sur '🔄 Reset APP', et on repart à zéro comme dans un bon jeu vidéo 🎮."
            }
            # Vérifier si la question fait partie de la FAQ
            return faq.get(user_input, "Nous n'avons pas de temps à perdre, allons à l'essentiel ! 🚀")

        # Afficher les messages déjà stockés
        display_messages()
        # Ajouter un champ de texte pour permettre à l'utilisateur de poser une question directement
        user_input = st.text_input("Posez votre propre question ici")
        if user_input:
            st.session_state['messages'].append({"role": "user", "content": user_input})

            with st.chat_message("user"):
                st.write(user_input)

            # Générer une réponse à partir de la FAQ
            response = faq_response(user_input)
            st.session_state['messages'].append({"role": "assistant", "content": response})

            # Afficher la réponse
            with st.chat_message("assistant"):
                st.write(response)

        # Si c'est la première interaction, répondre de manière drôle et sympathique
        if st.session_state['first_interaction']:
            welcome_message = first_interaction_response()
            st.session_state['messages'].append({"role": "assistant", "content": welcome_message})
            st.session_state['first_interaction'] = False

            with st.chat_message("assistant"):
                st.write(welcome_message)

        # Liste de questions prédéfinies
        questions = [
            "Comment importer des données ?",
            "Comment nettoyer les données ?",
            "Quels algorithmes de classification sont disponibles ?",
            "Comment explorer les données ?",
            "Qu'est-ce que le Playground ?",
            "Comment réinitialiser l'application ?"
        ]

        # Afficher les boutons de questions prédéfinies
        st.write("**Choisissez une question ci-dessous :**")
        col1, col2 = st.columns(2)

        with col1:
            for i in range(0, len(questions), 2):
                if st.button(questions[i]):
                    prompt = questions[i]
                    st.session_state['messages'].append({"role": "user", "content": prompt})

                    with st.chat_message("user"):
                        st.write(prompt)

                    # Générer une réponse à partir de la FAQ
                    response = faq_response(prompt)
                    st.session_state['messages'].append({"role": "assistant", "content": response})

                    # Afficher la réponse
                    with st.chat_message("assistant"):
                        st.write(response)

                    # Simuler un graphique si la question concerne les données
                    if "explorer les données" in prompt.lower() or "Playground" in prompt.lower():
                        with st.chat_message("assistant"):
                            st.line_chart(np.random.randn(30, 3))

        with col2:
            for i in range(1, len(questions), 2):
                if st.button(questions[i]):
                    prompt = questions[i]
                    st.session_state['messages'].append({"role": "user", "content": prompt})

                    with st.chat_message("user"):
                        st.write(prompt)

                    # Générer une réponse à partir de la FAQ
                    response = faq_response(prompt)
                    st.session_state['messages'].append({"role": "assistant", "content": response})

                    # Afficher la réponse
                    with st.chat_message("assistant"):
                        st.write(response)

                    # Simuler un graphique si la question concerne les données
                    if "explorer les données" in prompt.lower() or "Playground" in prompt.lower():
                        with st.chat_message("assistant"):
                            st.line_chart(np.random.randn(30, 3))


elif page == "README":
    st.markdown("# Readme")
    readme_content = read_readme()  # Lire le contenu du README
    st.markdown(readme_content)  # Afficher le contenu du README en Markdown
elif page == "Source de données":
    sourceData_page()  # Appel de la page pour l'import des données
elif page == "Nettoyage des données":
    nettoyageData_page()  # Appel de la page pour le nettoyage des données
elif page == "Aperçu du dataset":
    apercuData_page()  # Appel de la page pour l'aperçu des données
elif page == "Playground":
    type_ml = st.sidebar.radio(
        "Choisissez votre type de playground",
        ["Regression", "Classification", "NailsDetection"],
        index=None
    )

    if type_ml == "Regression":
        regression_page()
    elif type_ml == "Classification":
        classification_page()
    elif type_ml == "NailsDetection":
        nail_page()
    else:
        st.write("Choisissez une option")

# app.py, run with 'streamlit run app.py'
