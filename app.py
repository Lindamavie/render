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

    # Sauvegarde de l’image annotée
    output_path = os.path.join(tempfile.gettempdir(), "annotated_image.jpg")
    cv2.imwrite(output_path, overlay)

    return ("Altération détectée" if artifact_score > 0.03 else "Authentique"), output_path

# 📌 Génération d’un résumé explicatif détaillé
def generate_analysis_summary(ai_detection, artifact_check, annotated_image_url):
    summary = "📊 **Bilan de l'analyse de l'image**\n\n"
    
    if ai_detection == "Image IA":
        summary += "🔹 L'image semble avoir été **générée artificiellement** par une IA.\n"
    else:
        summary += "✅ L'image semble **authentique**, elle ne présente pas de signes évidents de génération par IA.\n"
    
    if artifact_check == "Altération détectée":
        summary += "⚠️ **Des modifications ont été détectées !**\n"
        summary += f"📌 Vous pouvez voir les zones retouchées sur [cette image annotée]({annotated_image_url}).\n"
    else:
        summary += "✅ Aucun signe de retouche ou d’altération majeure détecté.\n"
    
    summary += "\n📌 **Conclusion** : Cette image est " + (
        "peut-être falsifiée ou modifiée" if ai_detection == "Image IA" or artifact_check == "Altération détectée"
        else "probablement authentique"
    ) + "."
    
    return summary

# 📌 Route API principale : `/analyze`
@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    
    # Vérification du format
    if file.filename.split('.')[-1].lower() not in ['jpeg', 'jpg', 'png', 'bmp']:
        return jsonify({'error': 'Unsupported image format'}), 400
    
    filename = secure_filename(file.filename)
    file_path = os.path.join(tempfile.gettempdir(), filename)
    
    try:
        file.save(file_path)

        ela_image = error_level_analysis(file_path)
        ai_detection = detect_ai_generated(file_path)
        artifact_check, annotated_image_path = detect_and_visualize_artifacts(file_path)

        # Génération du lien de l’image annotée
        annotated_image_url = f"http://localhost:5000/annotated_image"

        # Génération du bilan d'analyse
        analysis_summary = generate_analysis_summary(ai_detection, artifact_check, annotated_image_url)

        return jsonify({
            'ai_detection': ai_detection,
            'artifact_check': artifact_check,
            'annotated_image_url': annotated_image_url,  # 🔥 Ajout du lien vers l’image annotée
            'analysis_summary': analysis_summary  
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

# 📌 Route pour télécharger l’image annotée
@app.route('/annotated_image', methods=['GET'])
def get_annotated_image():
    annotated_image_path = os.path.join(tempfile.gettempdir(), "annotated_image.jpg")
    if os.path.exists(annotated_image_path):
        return send_file(annotated_image_path, mimetype='image/jpeg')
    return jsonify({'error': 'Annotated image not found'}), 404

# Démarrer l'API Flask
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)