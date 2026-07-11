# Détection et Classification Automatique du Mélanome Cutané

Ce projet vise à concevoir un système intelligent basé sur le Deep Learning pour **détecter et classifier automatiquement le mélanome** à partir d'images dermatologiques. Le diagnostic précoce du mélanome améliore significativement la survie, et ce projet automatise l'analyse dermatoscopique, souvent chronophage et sujette aux erreurs.

---

## Fonctionnalités

*   **Prétraitement Médical Automatique :** Suppression des artefacts pileux et amélioration du contraste.
*   **Segmentation Précise :** Extraction de la lésion via un modèle **U-Net**.
*   **Classification Intelligente :** Évaluation de la probabilité de malignité (Bénin vs Malin) via **EfficientNetB0**.
*   **Interface Interactive :** Application web fluide développée avec **Streamlit** pour uploader et analyser les images en temps réel, incluant une visualisation clinique (superposition du masque).

---

## Architecture du Pipeline

L'application suit un pipeline structuré en 4 étapes majeures :

1.  **Prétraitement Clinique (Nettoyage des images)**
    *   **Suppression des poils :** Utilisation d'un filtre BlackHat et d'un seuillage binaire, suivi d'un algorithme d'Inpainting (TELEA) pour éliminer les poils sans altérer la lésion.
    *   **Amélioration du contraste (CLAHE) :** Application sur le canal de luminance (L) de l'espace colorimétrique LAB pour mieux révéler les structures pigmentaires.
2.  **Segmentation (U-Net)**
    *   Le modèle U-Net génère une carte de probabilités et isole la lésion (masque binaire).
    *   Recadrage précis de l'image sur la zone de la lésion détectée.
3.  **Classification (EfficientNetB0 - Transfer Learning)**
    *   Le modèle pré-entraîné sur ImageNet analyse la lésion isolée.
    *   Il prédit si le grain de beauté est **Malignant** ou **Benign**.
4.  **Déploiement**
    *   Affichage en 4 colonnes : Image prétraitée, Masque U-Net, Lésion isolée, et Diagnostic IA.

---

## Jeux de Données

Les modèles ont été entraînés sur deux bases de données distinctes issues de Kaggle :
*   **Modèle U-Net (Segmentation) :** Entraîné sur le dataset *HAM10000 Segmentation* (10 015 paires image/masque).
*   **Modèle EfficientNetB0 (Classification) :** Entraîné sur le *Melanoma Skin Cancer Dataset* (10 605 images, équilibrées lors des tests).

### Résultats et Métriques :
*   **Segmentation U-Net :** Dice Coefficient (Validation) de **0.890**.
*   **Classification EfficientNetB0 :** 
    *   AUC de Validation : **0.9734**
    *   Accuracy de Validation : **0.91**

---

## Installation & Exécution

### Prérequis
Assurez-vous d'avoir Python installé (recommandé : Python 3.9+). 
Vous pouvez utiliser un environnement virtuel.

### 1. Cloner ou télécharger le projet
```bash
# Placez-vous dans le répertoire du projet
cd melanoma
```

### 2. Installer les dépendances
Installez les bibliothèques requises à l'aide de `pip` :
```bash
pip install -r requirements.txt
```
*(Les dépendances incluent `streamlit`, `tensorflow`, `opencv-python`, `numpy`, `Pillow`)*

### 3. Lancer l'application
Démarrez l'interface web Streamlit avec la commande suivante :
```bash
streamlit run app.py
```
L'application s'ouvrira automatiquement dans votre navigateur par défaut (généralement à l'adresse `http://localhost:8501`).

---

## Technologies Utilisées

*   **Deep Learning :** TensorFlow, Keras
*   **Traitement d'Image :** OpenCV, Pillow (PIL)
*   **Manipulation des Données :** NumPy, Pandas
*   **Interface Web :** Streamlit
*   **Visualisation :** Matplotlib, Seaborn

