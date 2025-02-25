from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import Optional
import shutil
import os
import time
import uuid
import json
from datetime import datetime
from pydantic import BaseModel
import aiohttp

# Création de l'application FastAPI
app = FastAPI(title="Application de Sous-titrage Automatique")

# Configuration des dossiers statiques et templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Création des dossiers nécessaires s'ils n'existent pas
os.makedirs("uploads", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("stats", exist_ok=True)

# Fichier pour stocker les statistiques
STATS_FILE = "stats/statistics.json"

# Initialisation des statistiques si le fichier n'existe pas
if not os.path.exists(STATS_FILE):
    initial_stats = {
        "avg_processing_time": 0,
        "total_inferences": 0,
        "inferences_by_type": {
            "1": 0,  # Vidéo avec sous-titres intégrés
            "2": 0,  # Vidéo + sous-titres séparés
            "3": 0,  # Sous-titres uniquement
            "4": 0   # Texte uniquement
        },
        "pending_processes": 0,
        "pending_by_type": {
            "1": 0,
            "2": 0,
            "3": 0,
            "4": 0
        },
        "failed_processes": 0,
        "failed_by_type": {
            "1": 0,
            "2": 0,
            "3": 0,
            "4": 0
        },
        "processing_times": []  # Pour calculer la moyenne
    }
    with open(STATS_FILE, "w") as f:
        json.dump(initial_stats, f, indent=4)

# Fonction pour charger les statistiques
def load_statistics():
    try:
        with open(STATS_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        # En cas d'erreur, retourner les statistiques initiales
        return {
            "avg_processing_time": 0,
            "total_inferences": 0,
            "inferences_by_type": {"1": 0, "2": 0, "3": 0, "4": 0},
            "pending_processes": 0,
            "pending_by_type": {"1": 0, "2": 0, "3": 0, "4": 0},
            "failed_processes": 0,
            "failed_by_type": {"1": 0, "2": 0, "3": 0, "4": 0},
            "processing_times": []
        }

# Fonction pour sauvegarder les statistiques
def save_statistics(stats):
    with open(STATS_FILE, "w") as f:
        json.dump(stats, f, indent=4)

# Fonction pour mettre à jour les statistiques après un traitement
def update_statistics(request_type, processing_time, success=True):
    stats = load_statistics()
    
    request_type_str = str(request_type)
    
    if success:
        # Augmenter le nombre total d'inférences
        stats["total_inferences"] += 1
        stats["inferences_by_type"][request_type_str] += 1
        
        # Mettre à jour le temps de traitement moyen
        stats["processing_times"].append(processing_time)
        if len(stats["processing_times"]) > 100:  # Garder seulement les 100 derniers temps
            stats["processing_times"].pop(0)
        
        stats["avg_processing_time"] = sum(stats["processing_times"]) / len(stats["processing_times"])
        
        # Diminuer le nombre de processus en attente
        if stats["pending_processes"] > 0:
            stats["pending_processes"] -= 1
        if stats["pending_by_type"][request_type_str] > 0:
            stats["pending_by_type"][request_type_str] -= 1
    else:
        # Augmenter le nombre de processus échoués
        stats["failed_processes"] += 1
        stats["failed_by_type"][request_type_str] += 1
        
        # Diminuer le nombre de processus en attente
        if stats["pending_processes"] > 0:
            stats["pending_processes"] -= 1
        if stats["pending_by_type"][request_type_str] > 0:
            stats["pending_by_type"][request_type_str] -= 1
    
    save_statistics(stats)


import aiohttp  # Pour les requêtes HTTP asynchrones

# À l'intérieur de votre application FastAPI
async def process_video(file_path, request_type):
    """
    Traite le fichier en envoyant une requête à l'API du modèle d'IA.
    """
    # Créer l'ID du résultat
    result_id = str(uuid.uuid4())
    result_dir = f"results/{result_id}"
    os.makedirs(result_dir, exist_ok=True)
    
    # Déterminer le type de fichier
    file_ext = os.path.splitext(file_path)[1].lower()
    
    # Métriques pour les statistiques
    start_time = time.time()
    
    # Préparer le fichier pour envoi à l'API
    with open(file_path, "rb") as file:
        file_content = file.read()
    
    # Communiquer avec l'API du modèle d'IA de manière asynchrone
    async with aiohttp.ClientSession() as session:
        # Préparer les données pour l'API
        data = aiohttp.FormData()
        data.add_field("file", 
                      file_content, 
                      filename=os.path.basename(file_path),
                      content_type="video/mp4" if file_ext == ".mp4" else "audio/wav")
        data.add_field("type", str(request_type))
        
        # Envoyer la requête à l'API du modèle d'IA
        async with session.post("http://votre-api-ia/transcribe", data=data) as response:
            if response.status != 200:
                error_msg = await response.text()
                raise Exception(f"Erreur lors du traitement par l'API: {error_msg}")
            
            # Récupérer les résultats
            ia_results = await response.json()
    
    # Traiter les résultats reçus de l'API
    result_files = {}
    
    # Récupérer les sous-titres et texte depuis la réponse de l'API
    subtitles = ia_results.get("subtitles", "")
    text = ia_results.get("text", "")
    
    # Selon le type de requête, préparer les fichiers résultats
    if request_type == 1:  # Vidéo avec sous-titres intégrés
        # Récupérer la vidéo sous-titrée depuis l'API ou la générer localement
        if "video_with_subtitles" in ia_results:
            # Si l'API a fourni une vidéo sous-titrée
            result_video_path = f"{result_dir}/video_with_subtitles.mp4"
            with open(result_video_path, "wb") as f:
                f.write(await ia_results["video_with_subtitles"].read())
        else:
            # Sinon, utiliser un outil local pour incruster les sous-titres
            # (à implémenter)
            result_video_path = f"{result_dir}/video_with_subtitles.mp4"
            # ...code pour incruster les sous-titres...
            
        result_files["result_url"] = f"/results/{result_id}/video_with_subtitles.mp4"
        result_files["download_url"] = f"/download/{result_id}/video_with_subtitles.mp4"
    
    # Ajouter le code pour les autres types de requêtes...
    
    # Calculer le temps de traitement
    processing_time = time.time() - start_time
    
    return result_files, processing_time


# Route pour la page d'accueil
@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Route pour l'upload de vidéo
@app.post("/upload-video")
async def upload_video(
    video: UploadFile = File(...),
    type_of_request: int = Form(...),
):
    if not video.filename:
        raise HTTPException(status_code=400, detail="Aucun fichier n'a été fourni")
    
    # Vérifier le type de fichier
    file_ext = os.path.splitext(video.filename)[1].lower()
    if type_of_request in [1, 2] and file_ext != ".mp4":
        raise HTTPException(
            status_code=400,
            detail="Pour les options 1 et 2, seuls les fichiers MP4 sont acceptés"
        )
    
    if not (file_ext in [".mp4", ".wav"]):
        raise HTTPException(
            status_code=400,
            detail="Seuls les fichiers MP4 et WAV sont acceptés"
        )
    
    # Mettre à jour les statistiques - ajout d'un processus en attente
    stats = load_statistics()
    stats["pending_processes"] += 1
    stats["pending_by_type"][str(type_of_request)] += 1
    save_statistics(stats)
    
    # Créer un ID unique pour ce traitement
    processing_id = str(uuid.uuid4())
    
    # Créer le chemin du fichier temporaire
    file_path = f"uploads/{processing_id}{file_ext}"
    
    # Enregistrer le fichier uploadé
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
    except Exception as e:
        # En cas d'erreur, mettre à jour les statistiques
        update_statistics(type_of_request, 0, success=False)
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'enregistrement du fichier: {str(e)}")
    
    try:
        # Traiter le fichier
        start_time = time.time()
        result, processing_time = process_video(file_path, type_of_request)
        
        # Mettre à jour les statistiques - traitement réussi
        update_statistics(type_of_request, processing_time, success=True)
        
        return JSONResponse(content=result)
    
    except Exception as e:
        # En cas d'erreur, mettre à jour les statistiques
        update_statistics(type_of_request, 0, success=False)
        raise HTTPException(status_code=500, detail=f"Erreur lors du traitement: {str(e)}")
    
    finally:
        # Nettoyer le fichier temporaire
        if os.path.exists(file_path):
            os.remove(file_path)

# Route pour télécharger un résultat
@app.get("/download/{result_id}/{filename}")
async def download_result(result_id: str, filename: str):
    file_path = f"results/{result_id}/{filename}"
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Fichier non trouvé")
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="application/octet-stream"
    )

# Route pour accéder aux résultats
@app.get("/results/{result_id}/{filename}")
async def get_result(result_id: str, filename: str):
    file_path = f"results/{result_id}/{filename}"
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Fichier non trouvé")
    
    # Déterminer le type MIME en fonction de l'extension
    file_ext = os.path.splitext(filename)[1].lower()
    if file_ext == ".mp4":
        media_type = "video/mp4"
    elif file_ext == ".wav":
        media_type = "audio/wav"
    elif file_ext == ".srt":
        media_type = "text/plain"
    elif file_ext == ".txt":
        media_type = "text/plain"
    else:
        media_type = "application/octet-stream"
    
    return FileResponse(path=file_path, media_type=media_type)

# Route pour obtenir les statistiques
@app.get("/api/statistics")
async def get_statistics():
    stats = load_statistics()
    return {
        "avg_processing_time": stats["avg_processing_time"],
        "total_inferences": stats["total_inferences"],
        "inferences_by_type": stats["inferences_by_type"],
        "pending_processes": stats["pending_processes"],
        "pending_by_type": stats["pending_by_type"],
        "failed_processes": stats["failed_processes"],
        "failed_by_type": stats["failed_by_type"]
    }

# Point d'entrée pour uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)