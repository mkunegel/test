import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def apercuData_page():
    st.title("Aperçu du dataset 🎈")

    st.caption("Playground ML - Classification")

    # Vérifier si df est déjà chargé dans st.session_state
    if 'df' not in st.session_state:
        st.write("Aucune donnée n'a été chargée. Veuillez charger les données via la page Source de données.")

    else :
        df = st.session_state['df']  # Récupérer les données depuis st.session_state

        # Initialisation de selected_columns à une liste vide
        selected_columns = []

        # Permettre à l'utilisateur de saisir manuellement la colonne cible (y) via un text input
        target_input = st.text_input("Entrez le nom de la colonne cible (y)")

        if target_input:
            if target_input not in df.columns:
                st.error(f"La colonne '{target_input}' n'existe pas dans le dataset.")
            else:
                # Définir la colonne saisie comme variable cible (y)
                target = target_input
                st.write(f"La variable cible (y) est : {target}")
                # Stocker dans la session
                st.session_state['y'] = df[target_input]

                # Filtrer les colonnes numériques pour les variables explicatives (X) en excluant 'target'
                numeric_columns = [col for col in df.columns if
                                   pd.api.types.is_numeric_dtype(df[col]) and col != target]

                # Vérifier s'il y a des colonnes numériques disponibles
                if numeric_columns:
                    # Sélection des variables explicatives avec une infobulle
                    selected_columns = st.multiselect(
                        "Sélectionnez les variables explicatives (X)",
                        options=numeric_columns,
                        default=numeric_columns,
                        help="Les variables explicatives (X) doivent être numériques et influencent la variable cible (y)."
                    )

                    if selected_columns:
                        st.session_state['X'] = df[selected_columns]
                    else:
                        st.warning("Veuillez sélectionner au moins une variable explicative (X).")

                else:
                    st.warning("Aucune variable numérique disponible pour les variables explicatives (X).")
        else:
            st.warning("Veuillez entrer une colonne valide pour la variable cible (y).")

        # Création des sous-onglets
        subtab1, subtab2, subtab3 = st.tabs(
            ["Preview & Stats descriptives", "Matrice de corrélation & Pairplot", "ℹ variable du dataset"])

        # Sous-onglet 1 : Preview & Stats descriptives
        with subtab1:
            st.markdown("<h5 style='color: #FF5733; font-weight: bold;'>Dataset avant la sélection</h5>",
                        unsafe_allow_html=True)

            # Afficher le dataset entier avec des couleurs pour les colonnes X et y
            if st.session_state.get('X') is not None and st.session_state.get('y') is not None:
                styled_data = df.style.apply(
                    lambda x: ['background-color: lightblue' if col in st.session_state['X'].columns else
                               'background-color: lightgreen' if col == st.session_state['y'].name else ''
                               for col in df.columns], axis=1
                )
                st.dataframe(styled_data)

                # Utiliser HTML pour centrer l'image
                st.markdown(
                    """
                    <div style='text-align: center;'>
                        <img src="https://cdn-icons-png.flaticon.com/512/467/467262.png" alt="Example image" width="150">
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown("<h5 style='color: #FF5733; font-weight: bold;'>Dataset après la sélection (X et y)</h5>",
                            unsafe_allow_html=True)
                selected_data = df[st.session_state['X'].columns.tolist() + [st.session_state['y'].name]]
                styled_selected_data = selected_data.style.apply(
                    lambda x: ['background-color: lightblue' if col in st.session_state['X'].columns else
                               'background-color: lightgreen' if col == st.session_state['y'].name else ''
                               for col in selected_data.columns], axis=1
                )
                st.dataframe(styled_selected_data)

                st.markdown("---")  # Ligne de séparation

                # Résumé statistique des variables sélectionnées
                st.markdown(
                    "<h5 style='color: #FF5733; font-weight: bold;'>Résumé statistique des variables sélectionnées (X et y)</h5>",
                    unsafe_allow_html=True)
                st.write(selected_data.describe())

                st.markdown("---")  # Ligne de séparation

                # Visualisation des distributions des variables explicatives (X)
                st.markdown(
                    "<h6 style='color: #000000; font-weight: bold;'>Distributions des variables explicatives (X)</h6>",
                    unsafe_allow_html=True)

                # Affichage des histogrammes en 3 colonnes
                columns_per_row = 3
                for i, col in enumerate(st.session_state['X'].columns):
                    if i % columns_per_row == 0:
                        cols = st.columns(columns_per_row)

                    fig, ax = plt.subplots()
                    sns.histplot(st.session_state['X'][col], kde=True, ax=ax)
                    ax.set_title(f"Distribution de {col}")
                    cols[i % columns_per_row].pyplot(fig)

                st.markdown("---")  # Ligne de séparation

                # Visualisation des boxplots des variables explicatives (X) en fonction de la cible (y)
                st.markdown(
                    "<h6 style='color: #000000; font-weight: bold;'>Boxplots des variables explicatives (X) en fonction de la cible (y)</h6>",
                    unsafe_allow_html=True)

                # Affichage des boxplots en 3 colonnes
                for i, col in enumerate(st.session_state['X'].columns):
                    if i % columns_per_row == 0:
                        cols = st.columns(columns_per_row)

                    fig, ax = plt.subplots()
                    sns.boxplot(x=st.session_state['y'], y=col, data=selected_data, ax=ax)
                    ax.set_title(f"Boxplot de {col} en fonction de {st.session_state['y'].name}")
                    cols[i % columns_per_row].pyplot(fig)

        # Sous-onglet 2 : Matrice de corrélation & Pairplot
        with subtab2:
            # Titre de l'onglet
            st.markdown("<h5 style='color: #FF5733; font-weight: bold;'>Analyse des relations entre variables</h5>",
                        unsafe_allow_html=True)

            # Caption explicative pour cet onglet
            st.caption(
                "Dans cet onglet, nous analysons les relations entre les variables explicatives (X) à travers une matrice de corrélation et un pairplot. Cela permet d'identifier les relations linéaires et d'explorer visuellement les patterns entre variables.")

            # Titre avant la heatmap de corrélation
            st.markdown("<h6 style='color: #000000; font-weight: bold;'>Matrice de corrélation</h6>",
                        unsafe_allow_html=True)

            # Caption expliquant la matrice de corrélation
            st.caption(
                "La matrice de corrélation montre les coefficients de corrélation entre chaque paire de variables explicatives. Un coefficient proche de 1 indique une forte corrélation positive, tandis qu'un coefficient proche de -1 indique une forte corrélation négative.")

            # Vérifier que 'X' est bien initialisé et n'est pas None avant d'appeler la méthode corr()
            if st.session_state.get('X') is not None:
                # Assurez-vous que st.session_state['X'] est un DataFrame
                if not st.session_state['X'].empty:
                    corr_matrix = st.session_state['X'].corr()
                    # Afficher les données brutes dans un expander
                    with st.expander("Voir les données brutes de la matrice de corrélation"):
                        st.dataframe(corr_matrix)

                    # Afficher la heatmap de corrélation avec un titre
                    fig, ax = plt.subplots(figsize=(8, 4))
                    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
                    ax.set_title("Matrice de corrélation des variables explicatives (X)")
                    st.pyplot(fig)
                else:
                    st.warning("Le DataFrame X est vide, veuillez vérifier votre sélection des variables explicatives.")
            else:
                st.warning("Veuillez d'abord sélectionner les variables explicatives (X) dans l'onglet de sélection.")

            st.markdown("---")  # Ligne de séparation

            # Titre avant le pairplot
            st.markdown("<h6 style='color: #000000; font-weight: bold;'>Pairplot des variables explicatives (X)</h6>",
                        unsafe_allow_html=True)

            # Caption expliquant le pairplot
            st.caption(
                "Le pairplot visualise la relation entre chaque paire de variables explicatives. Il permet d'identifier des relations visuelles entre les variables, ainsi que des patterns potentiels comme la linéarité ou la dispersion.")

            # Vérifier que 'X' est bien un DataFrame valide avant d'appeler sns.pairplot()
            if st.session_state.get('X') is not None:
                # Assurez-vous que st.session_state['X'] est un DataFrame et non vide
                if isinstance(st.session_state['X'], pd.DataFrame) and not st.session_state['X'].empty:
                    # Afficher le pairplot
                    pairplot_fig = sns.pairplot(st.session_state['X'])
                    st.pyplot(pairplot_fig)
                else:
                    st.warning(
                        "Le DataFrame X est vide ou incorrect, veuillez vérifier votre sélection des variables explicatives.")
            else:
                st.warning("Veuillez d'abord sélectionner les variables explicatives (X) dans l'onglet de sélection.")

        # Sous-onglet 3 : Information sur les variables


        with subtab3:
            st.markdown("<h5 style='color: #FF5733; font-weight: bold;'>Description des variables ℹ (X et y)</h5>",
                        unsafe_allow_html=True)

            # Caption explicative pour ce sous-onglet
            st.caption(
                "Dans ce sous-onglet, nous affichons des informations détaillées sur chaque variable explicative (X) et la variable cible (y), qu'elle soit catégorielle ou numérique.")

            # Vérification si les variables explicatives (X) et la cible (y) sont disponibles
            if st.session_state.get('X') is not None and st.session_state.get('y') is not None:

                # Section pour les variables explicatives (X)
                st.markdown("<h6 style='color: #000000; font-weight: bold;'>Variables explicatives (X)</h6>",
                            unsafe_allow_html=True)

                # Tableau avec les statistiques importantes des variables explicatives (X)
                summary_data = []
                for col in st.session_state['X'].columns:
                    summary_data.append({
                        'Variable': col,
                        'Type': str(df[col].dtype),
                        'Valeurs uniques': df[col].nunique(),
                        'Min': df[col].min(),
                        'Max': df[col].max(),
                        'Moyenne': df[col].mean(),
                        'Écart-type': df[col].std()
                    })

                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df)

                # Séparateur avant la variable cible
                st.markdown("---")

                # Section pour la variable cible (y)
                st.markdown("<h6 style='color: #000000; font-weight: bold;'>Variable cible (y)</h6>",
                            unsafe_allow_html=True)
                target_col = st.session_state['y'].name

                # Vérifier si la variable cible est catégorielle ou numérique
                if pd.api.types.is_numeric_dtype(st.session_state['y']):
                    # Si y est numérique, afficher le tableau des statistiques numériques
                    y_summary = {
                        'Variable': target_col,
                        'Type': str(df[target_col].dtype),
                        'Valeurs uniques': df[target_col].nunique(),
                    }
                    st.dataframe(pd.DataFrame([y_summary]))

                else:
                    # Si y est catégorielle, afficher deux colonnes avec les valeurs textuelles et les fréquences
                    col1, col2 = st.columns(2)

                    with col1:
                        # Liste des catégories
                        st.markdown(f"**Liste des catégories de {target_col}** :")
                        st.write(df[target_col].unique().tolist())

                        # Tableau des fréquences des catégories
                        st.markdown(f"**Fréquence des catégories de {target_col}** :")
                        st.dataframe(df[target_col].value_counts())

                    with col2:
                        # Diagramme circulaire pour la répartition en pourcentage des catégories
                        fig, ax = plt.subplots(figsize=(6, 6))  # Taille ajustée pour une meilleure visualisation

                        # Calcul des pourcentages
                        category_percentages = df[target_col].value_counts(normalize=True) * 100

                        # Création du pie chart
                        wedges, texts, autotexts = ax.pie(category_percentages,
                                                          labels=category_percentages.index,
                                                          autopct='%1.1f%%',
                                                          colors=sns.color_palette('pastel'),
                                                          startangle=90,
                                                          wedgeprops={'edgecolor': 'black'})

                        # Style des textes et pourcentages
                        for text in texts:
                            text.set_fontsize(10)
                        for autotext in autotexts:
                            autotext.set_fontsize(10)

                        ax.set_title(f"Répartition en pourcentage de {target_col} (Pie Chart)", fontsize=10,
                                     fontweight='bold')
                        st.pyplot(fig)

            else:
                st.warning("Aucune sélection de variables explicatives (X) et de cible (y) n'a été effectuée.")
