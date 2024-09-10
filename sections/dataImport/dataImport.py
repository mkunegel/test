import streamlit as st
import pandas as pd


def sourceData_page():
    st.title("Source de données 🎈")

    df = pd.DataFrame()

    # Options pour choisir la base de données
    source_data = st.radio(
        "Choisissez votre source de données",
        ["vin.csv", "diabete.csv", "upload file (*.csv)"]
    )

    if source_data == "vin.csv":
        df = pd.read_csv("./data/vin.csv", index_col=0)
    elif source_data == "diabete.csv":
        df = pd.read_csv("./data/diabete.csv", index_col=0)
    elif source_data == "upload file (*.csv)":
        st.header("Importez vos données")

        separateur = st.text_input("Quel est le séparateur du fichier CSV ?",
                                   label_visibility='visible',
                                   disabled=False,
                                   placeholder=",")

        decimal = st.text_input("Quel est le décimal du fichier CSV ?",
                                   label_visibility='visible',
                                   disabled=False,
                                   placeholder=".")

        uploaded_file = st.file_uploader("Téléchargez votre fichier CSV", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file, sep=separateur, decimal=decimal)

    if not df.empty:
        st.subheader("Aperçu des données")
        st.write(df.head(5))
        if st.button("Enregistrer"):
            st.session_state['df'] = df
            st.success("Données enregistrées avec succès!")
    else:
        st.write("Veuillez choisir ou importer une source de données.")