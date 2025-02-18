from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np
from PIL import Image, ImageChops
import torch
import torchvision.transforms as transforms
import os
import tempfile
from werkzeug.utils import secure_filename
import logging

app = Flask(__name__)

# Configuration des logs
logging.basicConfig(level=logging.INFO)

# 📌 Détection des modifications via ELA (Erreur Niveau Analyse)
def error_level_analysis(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
        temp_path = os.path.join(tempfile.gettempdir(), "temp_ela.jpg")
        img.save(temp_path, 'JPEG', quality=90)
        ela_img = ImageChops.difference(img, Image.open(temp_path))
        os.remove(temp_path)
        return ela_img  # Retourne l’image ELA pour l’annotation
    except Exception as e:
        return str(e)

# 📌 Chargement du modèle IA pour détecter les images générées par IA
MODEL_PATH = "model_ia_detection.pth"
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
        model.eval()
    except Exception as e:
        logging.error(f"Erreur chargement modèle IA: {e}")

# 📌 Détection d’image générée par IA
def detect_ai_generated(image_path):
    if model is None:
        return "Modèle IA non chargé"
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
    
    prediction = torch.sigmoid(output).item()
    return "Image IA" if prediction > 0.5 else "Authentique"

# 📌 Détection d’altérations et génération d’une heatmap des zones modifiées
def detect_and_visualize_artifacts(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Détection des bords avec Canny
    edges = cv2.Canny(gray, 50, 150)

    # Mesure du pourcentage de pixels avec des artefacts
    artifact_score = np.sum(edges) / (gray.shape[0] * gray.shape[1])
    
    # Création d’une heatmap pour visualiser les zones retouchées
    heatmap = cv2.applyColorMap(edges, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.7, heatmap, 0.3, 0)

    # Affichage des images pour le débogage
    cv2.imshow("Heatmap", heatmap)
    cv2.imshow("Annotated Image", overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Sauvegarde de l’image annotée
    output_path = os.path.join(tempfile.gettempdir(), "annotated_image.jpg")
    success = cv2.imwrite(output_path, overlay)

    if not success:
        logging.error("❌ Erreur : Impossible de sauvegarder l’image annotée !")
        return "Erreur lors de la génération de l'image annotée", None
    else:
        logging.info(f"✅ Image annotée sauvegardée avec succès : {output_path}")
    
    return ("Altération détectée" if artifact_score > 0.03 else "Authentique"), output_path

# 📌 Génération d’un résumé explicatif détaillé
def generate_analysis_summary(ai_detection, artifact_check, annotated_image_url):
    summary = "📊 **Bilan de l'analyse de l'image**\n\n


