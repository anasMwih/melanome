import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Configuration de la page
st.set_page_config(page_title="Détection Mélanome IA", layout="wide", page_icon="🩺")

# Chargement des modèles (avec cache pour éviter le rechargement à chaque interaction)
@st.cache_resource
def load_models():
    try:
        # compile=False évite les erreurs liées aux métriques custom (DiceCoefficient)
        unet = tf.keras.models.load_model("models/unet_lesion_segmentation_final.keras", compile=False)
        classifier = tf.keras.models.load_model("models/EfficientNetB0_melanoma_classifier_final.keras", compile=False)
        return unet, classifier
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement des modèles : {e}")
        st.info("Vérifiez que les fichiers .keras sont dans le même dossier que app.py")
        return None, None

# Prétraitement médical
def medical_preprocessing(img_rgb):
    """
    Pipeline de nettoyage clinique :
    1. Suppression des poils par morphologie mathématique + inpainting
    2. Amélioration du contraste (CLAHE) pour révéler les structures pigmentaires
    """
    # 1. Détection & suppression des poils
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, hair_mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    img_clean = cv2.inpaint(img_rgb, hair_mask, 3, cv2.INPAINT_TELEA)

    # 2. Normalisation clinique (CLAHE sur le canal L de LAB)
    lab = cv2.cvtColor(img_clean, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    img_enhanced = cv2.merge([l, a, b])
    return cv2.cvtColor(img_enhanced, cv2.COLOR_LAB2RGB)

# Pipeline d'inférence complet
def run_inference(img_pil, unet, classifier, threshold=0.5):
    img_rgb = np.array(img_pil.convert("RGB"))
    
    # Redimensionnement pour EfficientNet (300x300)
    img_resized = cv2.resize(img_rgb, (300, 300))
    img_processed = medical_preprocessing(img_resized)

    # Segmentation U-Net (attend 256x256, normalisé [0,1])
    img_unet = cv2.resize(img_processed, (256, 256)).astype(np.float32) / 255.0
    mask_pred = unet.predict(np.expand_dims(img_unet, axis=0), verbose=0)[0]
    mask_binary = (mask_pred > threshold).astype(np.uint8) * 255
    mask_resized = cv2.resize(mask_binary, (300, 300), interpolation=cv2.INTER_NEAREST)

    # Isolation & Recadrage de la lésion
    lesion = cv2.bitwise_and(img_processed, img_processed, mask=mask_resized)
    coords = cv2.findNonZero(mask_resized)
    
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        lesion_crop = lesion[y:y+h, x:x+w]
        lesion_crop = cv2.resize(lesion_crop, (300, 300))
    else:
        lesion_crop = img_processed  # Fallback si aucun masque détecté

    #  Classification EfficientNetB0
    img_class = tf.keras.applications.efficientnet.preprocess_input(lesion_crop.astype(np.float32))
    pred = classifier.predict(np.expand_dims(img_class, axis=0), verbose=0)[0][0]
    prob_mal = float(pred)
    prob_ben = 1.0 - prob_mal

    return {
        "original": img_processed,
        "mask": mask_resized,
        "lesion": lesion_crop,
        "label": "Malignant" if prob_mal > 0.5 else "Benign",
        "prob_mal": prob_mal,
        "prob_ben": prob_ben
    }

# Interface Streamlit
def main():
    st.title("🩺 Détection & Classification Automatique du Mélanome Cutané")
    st.markdown("Projet 4ème année IASD7 | Deep Learning ")

    with st.sidebar:
        st.header("⚙️ Paramètres")
        threshold = st.slider("Seuil de segmentation (U-Net)", 0.3, 0.7, 0.5, 0.05)
        st.info("💡 Formats acceptés : JPG, JPEG, PNG\n📏 Résolution recommandée : ≥300x300px")
        st.divider()
        

    uploaded_file = st.file_uploader("📤 Uploader une image dermatoscopique", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        with st.spinner("🔄 Chargement des modèles & analyse en cours..."):
            unet, classifier = load_models()
            if unet is None or classifier is None:
                st.stop()

            img_pil = Image.open(uploaded_file)
            results = run_inference(img_pil, unet, classifier, threshold=threshold)

        # Affichage des résultats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.subheader("📷 Image Prétraitée")
            st.image(results["original"], use_container_width=True)
        with col2:
            st.subheader("🥽 Masque U-Net")
            st.image(results["mask"], use_container_width=True)
        with col3:
            st.subheader("🔍 Lésion Isolée")
            st.image(results["lesion"], use_container_width=True)
        with col4:
            st.subheader("📊 Diagnostic IA")
            if results["label"] == "Malignant":
                st.error(f"⚠️ **{results['label']}**")
            else:
                st.success(f"✅ **{results['label']}**")
            st.metric("Probabilité Malin", f"{results['prob_mal']:.2%}")
            st.metric("Probabilité Bénin", f"{results['prob_ben']:.2%}")

        # Visualisation clinique superposée
        st.divider()
        st.subheader("🩻 Visualisation Clinique (Superposition Masque/Image)")
        overlay = results["original"].copy()
        mask_color = cv2.cvtColor(results["mask"], cv2.COLOR_GRAY2RGB)
        mask_red = np.zeros_like(overlay)
        mask_red[mask_color[:,:,0] == 255] = [255, 0, 0]
        overlay = cv2.addWeighted(overlay, 0.7, mask_red, 0.5, 0)
        st.image(overlay, use_container_width=True)

        # Note technique
        with st.expander("Détails techniques du pipeline"):
            st.markdown("""
            1. **Prétraitement** : Suppression des poils (BlackHat + Inpainting) + Normalisation CLAHE
            2. **Segmentation** : U-Net → Masque binaire → Recadrage sur la lésion
            3. **Classification** : EfficientNetB0 (Transfer Learning) sur la lésion isolée
            4. **Métriques** : Dice Coefficient (U-Net) | AUC & Accuracy (EfficientNet)
            """)

if __name__ == "__main__":
    main()