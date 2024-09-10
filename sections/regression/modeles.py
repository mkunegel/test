import streamlit as st
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import lasso_path
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from lazypredict.Supervised import LazyRegressor
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb

# Fonction principale pour l'affichage des modeles
def show_modeles(X_train, X_test, y_train, y_test, random_state):
    # Séparation des onglets pour chaque modèle
    lazy_tab, lasso_tab, linear_regression_tab, extra_trees_tab, xgboost_tab = st.tabs(
        ["LazyRegressor", "Régression Lasso", "Régression Linéaire", "Extra Trees Regressor", "XGBoost"])

    # Appel de chaque modèle dans son onglet
    with lazy_tab:
        show_lazy_regressor(X_train, X_test, y_train, y_test)

    with lasso_tab:
        show_lasso_model(X_train, X_test, y_train, y_test)

    with linear_regression_tab:
        show_linear_regression(X_train, X_test, y_train, y_test)

    with extra_trees_tab:
        show_extra_trees(X_train, X_test, y_train, y_test, random_state)

    with xgboost_tab:
        show_xgboost(X_train, X_test, y_train, y_test, random_state)

# Fonction pour l'affichage du learning curve cbof
def plot_learning_curve(model, X, y):
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=5, scoring='neg_mean_squared_error',
                                                            n_jobs=-1)
    train_mean = -np.mean(train_scores, axis=1)
    test_mean = -np.mean(test_scores, axis=1)

    fig, ax = plt.subplots()
    ax.plot(train_sizes, train_mean, label='Erreur d\'Entraînement', marker='o')
    ax.plot(train_sizes, test_mean, label='Erreur de Validation', marker='o')
    ax.set_xlabel('Taille de l\'échantillon d\'apprentissage')
    ax.set_ylabel('Erreur Quadratique Moyenne')
    ax.set_title('Courbe d\'Apprentissage')
    ax.legend()

    return fig

# Fonction pour l'affichage du LazyRegressor
def show_lasso_model(X_train, X_test, y_train, y_test):
    # Initialisation des variables de session
    if 'grid_search' not in st.session_state:
        st.session_state.grid_search = None
        st.session_state.best_alpha = None
        st.session_state.selected_alpha = None

    st.caption("Le Gridsearch permet de déterminer les meilleurs hyperparamètres d'alpha pour le modèle de regréssion Lasso.")

    # Bouton pour lancer le GridSearch
    if st.button("Lancer un GridSearch"):
        with st.spinner("Recherche des hyperparamètres en cours..."):
            param_grid = {'alpha': np.logspace(-4, 4, 100)}
            lasso = Lasso()
            grid_search = GridSearchCV(lasso, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            st.session_state.grid_search = grid_search
            st.session_state.best_alpha = grid_search.best_params_['alpha']
            st.session_state.selected_alpha = st.session_state.best_alpha
        st.success(f"Recherche terminée! Meilleure Alpha: {st.session_state.best_alpha}")

    if st.session_state.grid_search is not None and st.session_state.best_alpha is not None:
        # Explication du paramètre alpha
        st.markdown("""
        **Alpha (λ)** est un hyperparamètre de régularisation utilisé dans la régression Lasso. 

        - **Régularisation** : La régression Lasso ajoute une pénalité à la somme des valeurs absolues des coefficients de régression. Cette pénalité a pour effet de réduire certains coefficients à zéro, ce qui peut aider à la sélection de variables et à la prévention du sur-apprentissage (overfitting).

        - **Valeur d'alpha élevée** : Une valeur élevée de alpha augmente la pénalité, ce qui force les coefficients à se rapprocher de zéro. Cela peut simplifier le modèle mais peut aussi conduire à un sous-ajustement (underfitting).

        - **Valeur d'alpha faible** : Une valeur faible de alpha diminue la pénalité, permettant aux coefficients de rester plus grands. Cela peut rendre le modèle plus flexible mais peut également conduire à un sur-ajustement si le modèle est trop complexe.

        En résumé, le choix de alpha est crucial pour équilibrer la complexité du modèle et la capacité de généralisation. Vous pouvez tester différentes valeurs pour voir comment cela impacte la performance du modèle sur vos données.
        """)

        # Input pour modifier l'alpha au choix
        alpha_input = st.text_input(
            'Entrez une valeur d\'alpha',
            value=str(round(st.session_state.best_alpha, 4)),
            help="Alpha est un paramètre de régularisation dans la régression Lasso. Il contrôle la force de la pénalité appliquée aux coefficients du modèle. Plus la valeur est élevée, plus les coefficients seront contraints à zéro. Une bonne valeur d'alpha permet d'obtenir un bon équilibre entre sous-ajustement et sur-ajustement."
        )

        try:
            alpha = float(alpha_input)
            if alpha <= 0:
                st.error("L'alpha doit être un nombre positif.")
            else:
                if st.button("Évaluer Régression Lasso"):
                    lasso = Lasso(alpha=alpha)
                    lasso.fit(X_train, y_train)
                    y_pred = lasso.predict(X_test)
                    mse_lasso = mean_squared_error(y_test, y_pred)
                    r2_lasso = r2_score(y_test, y_pred)
                    # Affichage des scores
                    st.write(f"**Alpha utilisé pour l'évaluation :** {alpha}")
                    st.write(f"**MSE (Erreur Quadratique Moyenne) :** {mse_lasso:.2f}")
                    st.write(f"**R^2 Score (Coefficient de Détermination) :** {r2_lasso:.2f}")
                    # Affichage de la comparaison entre la prédiction et la target dans un expander
                    comparaison = pd.DataFrame({'Valeur Réelle (y_test)': y_test, 'Valeur Prédite (y_pred)': y_pred})
                    with st.expander("Comparaison entre la prédiction et la target"):
                        st.dataframe(comparaison)

                    # Séparation en colonne puis affichage des graphs
                    col1, col2 = st.columns(2)

                    with col1:
                        residuals = y_test - y_pred  # Calcul des résidus
                        fig, ax = plt.subplots()  # Création de la figure
                        ax.scatter(y_pred, residuals, alpha=0.5)  # Nuage de points (valeurs prédites vs résidus)
                        ax.axhline(y=0, color='red', linestyle='--')  # Ligne rouge indiquant 0 (résidu nul)
                        ax.set_xlabel('Valeurs Prédites (y_pred)')  # Etiquette pour l'axe des X
                        ax.set_ylabel('Résidus')  # Etiquette pour l'axe des Y
                        ax.set_title('Graphique des Résidus pour Régression Lasso', loc='center')  # Titre du graphique
                        st.pyplot(fig)  # Affichage du graphique avec Streamlit

                    with col2:
                        fig, ax = plt.subplots()

                        # Scatter plot des valeurs observées vs prédictions
                        ax.scatter(y_test, y_pred, alpha=0.5, label='Données')

                        # Ajout d'une ligne de régression (y=x) pour indiquer un ajustement parfait
                        ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--',
                                label='y = x (ajustement parfait)')

                        # Mise en forme du graphique
                        ax.set_xlabel('Valeurs Réelles (y_test)')
                        ax.set_ylabel('Valeurs Prédites (y_pred)')
                        ax.set_title('Droite de Régression pour Régression Lasso')
                        ax.legend()

                        st.pyplot(fig)  # Affichage du graphique dans Streamlit
        except ValueError:
            st.error("Veuillez entrer une valeur numérique valide pour alpha.")
    else:
        st.info("Cliquez sur 'Lancer GridSearchCV' pour commencer la recherche des hyperparamètres.")

# Fonction pour la régression lineaire
def show_linear_regression(X_train, X_test, y_train, y_test):
    st.caption("Régression Linéaire")

    st.info(
        """La régression linéaire est une méthode utilisée pour prédire une valeur en fonction d'une autre. Elle cherche à tracer une ligne droite (ou un plan pour plusieurs variables) qui représente le mieux la relation entre les variables. Cela nous aide à comprendre comment les variables sont liées et à faire des prévisions."""
    )


    if st.button("Évaluer Régression Linéaire"):
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        comparaison = pd.DataFrame({
            'Valeur Réelle (y_test)': y_test,
            'Valeur Prédite (y_pred)': y_pred
        })

        # Calcul et affichage des scores
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.write(f"**MSE (Erreur Quadratique Moyenne) :** {mse:.2f}")
        st.write(f"**R^2 Score (Coefficient de Détermination) :** {r2:.2f}")

        with st.expander("Comparaison entre la prédiction et la target"):
            st.dataframe(comparaison)

        # Séparation en colonnes puis affichage des graphiques
        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots()
            ax.scatter(comparaison['Valeur Réelle (y_test)'], comparaison['Valeur Prédite (y_pred)'], alpha=0.5,
                       color='blue', label='Prédictions')
            ax.plot([min(comparaison['Valeur Réelle (y_test)']), max(comparaison['Valeur Réelle (y_test)'])],
                    [min(comparaison['Valeur Réelle (y_test)']), max(comparaison['Valeur Réelle (y_test)'])],
                    color='red', linestyle='--', label='Référence')
            ax.set_xlabel('Valeurs Réelles (y_test)')
            ax.set_ylabel('Valeurs Prédites (y_pred)')
            ax.set_title('Graphique de Dispersion', loc='center')
            ax.legend()
            st.pyplot(fig)

        with col2:
            fig, ax = plt.subplots()
            sns.regplot(x='Valeur Réelle (y_test)', y='Valeur Prédite (y_pred)', data=comparaison,
                        scatter_kws={'alpha': 0.5, 'color': 'blue'}, line_kws={'color': 'red'}, ax=ax)
            ax.set_xlabel('Valeurs Réelles (y_test)')
            ax.set_ylabel('Valeurs Prédites (y_pred)')
            ax.set_title('Graphique de Régression', loc='center')
            st.pyplot(fig)

# Fonction pour l'extra trees regressor
def show_extra_trees(X_train, X_test, y_train, y_test, random_state):
    st.caption("Extra Trees Regressor")

    # Description du modèle
    st.info(
        """Extra Trees Regressor est un modèle qui utilise plusieurs arbres de décision pour faire des prévisions. Il prend la moyenne des prédictions de tous les arbres pour donner une estimation finale. Ce modèle est efficace pour capturer des relations complexes dans les données et peut améliorer les prédictions par rapport à un seul arbre de décision."""
    )

    # Paramètres du modèle avec aide contextuelle
    n_estimators = st.slider(
        "Nombre d'arbres (n_estimators)",
        min_value=10,
        max_value=500,
        value=100,
        step=10,
        key="extra_trees_n_estimators",
        help="Le nombre d'arbres dans la forêt. Plus il y a d'arbres, plus le modèle peut capturer des complexités. Cependant, un trop grand nombre d'arbres peut augmenter le temps de calcul sans forcément améliorer les performances du modèle."
    )

    max_depth = st.slider(
        "Profondeur maximale (max_depth)",
        min_value=1,
        max_value=40,
        value=10,
        key="extra_trees_max_depth",
        help="La profondeur maximale des arbres. Une profondeur plus grande permet aux arbres d'apprendre des relations plus complexes dans les données, mais peut aussi conduire à un sur-apprentissage. Une profondeur plus petite limite la complexité du modèle et peut éviter le sur-apprentissage."
    )

    bootstrap = st.selectbox(
        "Utiliser Bootstrap ?",
        [False, True],
        key="extra_trees_bootstrap",
        help="Détermine si les échantillons pour chaque arbre doivent être tirés avec ou sans remise. Lorsque le bootstrap est activé, chaque arbre est entraîné sur un échantillon aléatoire avec remise des données d'entraînement, ce qui peut améliorer la généralisation du modèle."
    )

    # Bouton pour lancer le modèle
    if st.button("Évaluer Extra Trees Regressor"):
        model = ExtraTreesRegressor(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap,
                                    random_state=random_state)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Calcul et affichage des scores
        comparaison = pd.DataFrame({
            'Valeur Réelle (y_test)': y_test,
            'Valeur Prédite (y_pred)': y_pred
        })

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.subheader("Évaluation du Modèle Extra Trees Regressor")
        st.write(f"**MSE (Erreur Quadratique Moyenne) :** {mse:.2f}")
        st.write(f"**R^2 Score (Coefficient de Détermination) :** {r2:.2f}")

        with st.expander("Comparaison entre la prédiction et la target"):
            st.dataframe(comparaison)

        # Séparation en colonnes puis affichage des graphiques
        col1, col2 = st.columns(2)

        with col1:
            fig = plot_learning_curve(model, X_train, y_train)
            st.pyplot(fig)

        with col2:
            feature_importances = model.feature_importances_
            features = X_train.columns
            fig, ax = plt.subplots()
            sns.barplot(x=feature_importances, y=features, ax=ax)
            ax.set_title('Importance des Caractéristiques')
            st.pyplot(fig)

        st.write("""
        Le BMI (Indice de Masse Corporelle) et le S5 (logarithme des triglycérides) se révèlent être les caractéristiques les plus influentes.

        Un BMI élevé est un indicateur majeur d'obésité, souvent associée au diabète de type 2. L'obésité est liée à des déséquilibres métaboliques qui augmentent le risque de développer le diabète.

        De même, des niveaux élevés de triglycérides (capturés par S5) sont fréquemment liés à des déséquilibres métaboliques et cardiovasculaires. Ces déséquilibres sont étroitement associés au diabète, car ils reflètent des anomalies dans la gestion des graisses par le corps.
        """)


# Fonction pour l'xgboost
def show_xgboost(X_train, X_test, y_train, y_test, random_state):
    st.caption("XGBoost Regressor")

    # Description du modèle
    st.info(
        """XGBoost Regressor est un modèle de boosting qui construit plusieurs arbres de décision pour améliorer les prévisions. Chaque arbre aide à corriger les erreurs des arbres précédents. Il est apprécié pour sa rapidité et ses bonnes performances, même avec des relations complexes et de grandes quantités de données."""
    )

    # Paramètres du modèle avec aide contextuelle
    n_estimators = st.slider(
        "Nombre d'arbres (n_estimators)",
        min_value=10,
        max_value=500,
        value=100,
        step=10,
        key="xgboost_n_estimators",
        help="Le nombre total d'arbres que le modèle construira. Plus il y a d'arbres, plus le modèle peut capturer des relations complexes, mais cela peut aussi augmenter le temps de calcul."
    )

    max_depth = st.slider(
        "Profondeur maximale (max_depth)",
        min_value=1,
        max_value=20,
        value=6,
        key="xgboost_max_depth",
        help="La profondeur maximale des arbres. Une profondeur plus grande permet aux arbres d'apprendre des relations plus complexes, mais peut aussi entraîner un sur-ajustement."
    )

    learning_rate = st.slider(
        "Taux d'apprentissage (learning_rate)",
        min_value=0.01,
        max_value=0.3,
        value=0.1,
        step=0.01,
        key="xgboost_learning_rate",
        help="Le taux d'apprentissage ajuste la contribution de chaque arbre à la prédiction finale. Un taux plus bas rend l'apprentissage plus lent mais peut améliorer la précision."
    )

    gamma = st.slider(
        "Gamma",
        min_value=0.0,
        max_value=5.0,
        value=0.0,
        step=0.1,
        key="xgboost_gamma",
        help="Le paramètre gamma contrôle la régularisation. Il définit le minimum de réduction de la perte nécessaire pour faire une nouvelle séparation dans un arbre."
    )

    subsample = st.slider(
        "Subsample",
        min_value=0.1,
        max_value=1.0,
        value=1.0,
        step=0.1,
        key="xgboost_subsample",
        help="La proportion des échantillons utilisés pour construire chaque arbre. Un sous-échantillonnage peut aider à éviter le sur-ajustement en ajoutant de la diversité."
    )

    # Bouton pour lancer le modèle
    if st.button("Évaluer XGBoost Regressor"):
        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            gamma=gamma,
            subsample=subsample,
            random_state=random_state
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Calcul et affichage des scores
        comparaison = pd.DataFrame({
            'Valeur Réelle (y_test)': y_test,
            'Valeur Prédite (y_pred)': y_pred
        })

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.write(f"**MSE (Erreur Quadratique Moyenne) :** {mse:.2f}")
        st.write(f"**R^2 Score (Coefficient de Détermination) :** {r2:.2f}")

        with st.expander("Comparaison entre la prédiction et la target"):
            st.dataframe(comparaison)

        # Séparation en colonnes puis affichage des graphiques
        col1, col2 = st.columns(2)

        with col1:
            fig = plot_learning_curve(model, X_train, y_train)
            st.pyplot(fig)

        with col2:
            feature_importances = model.feature_importances_
            features = X_train.columns
            fig, ax = plt.subplots()
            sns.barplot(x=feature_importances, y=features, ax=ax)
            ax.set_title('Importance des Caractéristiques')
            st.pyplot(fig)

        st.write("""
                Le BMI (Indice de Masse Corporelle) et le S5 (logarithme des triglycérides) se révèlent être les caractéristiques les plus influentes.

                Un BMI élevé est un indicateur majeur d'obésité, souvent associée au diabète de type 2. L'obésité est liée à des déséquilibres métaboliques qui augmentent le risque de développer le diabète.

                De même, des niveaux élevés de triglycérides (capturés par S5) sont fréquemment liés à des déséquilibres métaboliques et cardiovasculaires. Ces déséquilibres sont étroitement associés au diabète, car ils reflètent des anomalies dans la gestion des graisses par le corps.
                """)

# Fonction principale pour afficher le Lazy Regressor
def show_lazy_regressor(X_train, X_test, y_train, y_test):
    st.caption("LazyRegressor")
    st.info("Le LazyRegressor permet de tester plusieurs modèles et identifier ceux qui s'adaptent le mieux à nos données.")

    # Bouton pour lancer LazyRegressor
    if st.button("Lancer LazyRegressor"):
        with st.spinner('Calcul des performances des modèles en cours...'):
            reg = LazyRegressor()
            models, predictions = reg.fit(X_train, X_test, y_train, y_test)

        st.success("Modèles évalués avec succès!")
        st.dataframe(models)

        # Interprétation basique des résultats
        best_model = models.index[0]
        st.write(
            f"Le modèle qui semble le plus performant est : **{best_model}** avec un coefficient de détermination (R²) de {models.loc[best_model, 'R-Squared']}.")

        # Explications des métriques
        st.subheader("Explication des métriques")
        st.write("""
            - **R² (Coefficient de détermination)** : Mesure la proportion de variance expliquée par le modèle. Un R² de 1 signifie que le modèle explique parfaitement les données, tandis qu'un R² de 0 indique que le modèle n'explique aucune variance.
            - **R² ajusté** : Comme le R², mais ajusté pour le nombre de prédicteurs dans le modèle. Il pénalise les modèles qui utilisent trop de variables non pertinentes.
            - **RMSE (Root Mean Squared Error)** : Indique la racine carrée de l'erreur quadratique moyenne. Plus la RMSE est faible, plus les prédictions du modèle sont proches des valeurs réelles.
        """)

        # Préparation des noms des modèles pour éviter la coupure
        max_length = 30
        models.index = [name if len(name) <= max_length else name[:max_length-3] + '...' for name in models.index]

        # Affichage sous forme de DataFrame des 5 meilleurs modèles
        st.subheader("Top 5 des meilleurs modèles :")
        top_5 = models.head(5)[['R-Squared', 'Adjusted R-Squared', 'RMSE']]
        st.dataframe(top_5)

        # Comparaison des 5 meilleurs modèles avec des graphiques dans deux colonnes
        st.subheader("Comparaison des 5 meilleurs modèles")

        # Création des colonnes
        col1, col2 = st.columns(2)

        # Graphique de comparaison des RMSE dans la première colonne
        with col1:
            fig1, ax1 = plt.subplots()
            sns.barplot(x=top_5.index, y=top_5['RMSE'], ax=ax1)
            for i, value in enumerate(top_5['RMSE']):
                ax1.text(i, value, round(value, 2), ha='center', va='bottom')
            ax1.set_title("Comparaison des RMSE")
            ax1.set_ylabel("RMSE")
            ax1.set_xlabel("Modèles")
            ax1.tick_params(axis='x', rotation=45)  # Rotation des labels de l'axe x
            st.pyplot(fig1)

        # Graphique de comparaison des R² dans la deuxième colonne
        with col2:
            fig2, ax2 = plt.subplots()
            sns.barplot(x=top_5.index, y=top_5['R-Squared'], ax=ax2)
            for i, value in enumerate(top_5['R-Squared']):
                ax2.text(i, value, round(value, 4), ha='center', va='bottom')
            ax2.set_title("Comparaison des R²")
            ax2.set_ylabel("R²")
            ax2.set_xlabel("Modèles")
            ax2.tick_params(axis='x', rotation=45)  # Rotation des labels de l'axe x
            st.pyplot(fig2)