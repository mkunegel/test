import streamlit as st
from roboflow import Roboflow
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import os

# Initialiser Roboflow avec la clé API
rf = Roboflow(api_key="N5DG8Y61gDw4qE31kXBa")
project = rf.workspace().project("dignamic-ml")
model = project.version(1).model

def predire_image(image, conf):
    # Sauvegarde temporaire de l'image
    temp_img_path = "temp_image.jpg"
    image.save(temp_img_path)

    # Utiliser le modèle Roboflow pour faire la prédiction
    result = model.predict(temp_img_path, confidence=conf).json()
    return result

def draw_predictions(image, predictions):
    draw = ImageDraw.Draw(image)

    # Essayer de charger une police par défaut, sinon utiliser les paramètres par défaut
    try:
        font = ImageFont.load_default()
    except Exception as e:
        font = None
        print(f"Erreur lors du chargement de la police : {e}")

    for prediction in predictions.get('predictions', []):
        # Extraire les coordonnées des boîtes de détection
        x_min = int(prediction['x'] - prediction['width'] / 2)
        y_min = int(prediction['y'] - prediction['height'] / 2)
        x_max = int(prediction['x'] + prediction['width'] / 2)
        y_max = int(prediction['y'] + prediction['height'] / 2)

        # Dessiner le rectangle autour de la détection
        draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)

        # Préparer le texte pour l'étiquette
        label = prediction['class']
        confidence = prediction.get('confidence', 0)  # Obtenir la confiance, défaut à 0 si non disponible
        confidence_text = f"{confidence * 100:.1f}%"  # Convertir en pourcentage

        # Préparer le texte complet
        text = f"{label} ({confidence_text})"

        # Utiliser une taille de police proportionnelle à la taille de la boîte de détection
        text_width = x_max - x_min
        text_height = y_max - y_min
        if font:
            # Ajuster la taille de la police pour s'adapter à la boîte de détection
            font_size = min(text_width, text_height) // 10
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except Exception as e:
                font = ImageFont.load_default()
                print(f"Erreur lors du chargement de la police TTF : {e}")

            # Calculer la taille du texte avec textbbox
            bbox = draw.textbbox((x_min, y_min), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        else:
            text_width = 0
            text_height = 0

        # Dessiner un fond coloré pour le texte
        background_x0 = x_min
        background_y0 = y_min - text_height - 5
        background_x1 = x_min + text_width
        background_y1 = y_min

        # Dessiner un rectangle en arrière-plan pour améliorer la lisibilité du texte
        draw.rectangle([background_x0, background_y0, background_x1, background_y1], fill="black")

        # Dessiner le texte avec les paramètres de police par défaut
        shadow_offset = 1
        if font:
            draw.text((x_min + shadow_offset, y_min - text_height - 5 + shadow_offset), text, fill="white", font=font)
            draw.text((x_min, y_min - text_height - 5), text, fill="red", font=font)
        else:
            draw.text((x_min + shadow_offset, y_min - text_height - 5 + shadow_offset), text, fill="white")
            draw.text((x_min, y_min - text_height - 5), text, fill="red")

    return image

def results(result):
    """Affiche les résultats dans un DataFrame."""
    predictions = result.get('predictions', [])
    if predictions:
        # Convertir les résultats en DataFrame
        df = pd.DataFrame(predictions)
        st.write("**Résultats de la détection :**")
        st.dataframe(df)
    else:
        st.write("Aucune détection trouvée.")

    with st.expander("Afficher les détails JSON"):
        st.json(result)

def nail_page():
    st.caption("Bienvenue dans le Playground de la détection d'ongle")
    st.subheader("Détection d'ongles")
    st.info("Téléchargez une image ou choisissez une image de test pour détecter les ongles.")

    # Chemin vers le fichier image test
    test_image_path = 'data/ongles1.jpeg'

    # Ajouter une option pour choisir l'image test ou télécharger une image
    option = st.selectbox(
        "Choisissez une option",
        ["Téléchargez une image", "Image de test"]
    )

    image = None

    if option == "Téléchargez une image":
        # Charger l'image depuis l'utilisateur
        uploaded_image = st.file_uploader("Choisissez une image...", type="jpg")

        if uploaded_image is not None:
            # Afficher l'image téléchargée
            image = Image.open(uploaded_image)
    else:
        # Charger l'image test choisie
        if os.path.exists(test_image_path):
            image = Image.open(test_image_path)
        else:
            st.error(f"L'image de test '{test_image_path}' est introuvable.")
            return

    if image:
        # Afficher l'image sélectionnée
        with st.expander("Afficher l'image"):
            st.image(image, caption='Image sélectionnée', use_column_width=True)

        # Demander l'intervalle de confiance
        conf = st.slider("Sélectionnez l'intervalle de confiance", 0, 100, 50)
        st.write("**Intervalle de confiance:**", conf)

        # Ajouter un bouton pour lancer la prédiction
        if st.button("Lancer la prédiction"):
            with st.spinner("Analyse de l'image..."):
                result = predire_image(image, conf)

            # Afficher les résultats dans un DataFrame
            results(result)

            # Dessiner les prédictions sur l'image
            if 'predictions' in result:
                annotated_image = draw_predictions(image.copy(), result)
                st.image(annotated_image, caption='Image avec détections', use_column_width=True)
    else:
        st.write("Veuillez sélectionner une image pour continuer.")

