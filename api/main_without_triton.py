import base64
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
from transformers import pipeline
import torch

# Initialiser le modèle Whisper une seule fois au démarrage
device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model = pipeline("automatic-speech-recognition",
                        "openai/whisper-small.en",  # Modèle plus petit pour des temps de traitement plus rapides
                        torch_dtype=torch.float32,
                        device=device,
                        return_timestamps=True)


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
            "4": 0  # Texte uniquement
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

@app.get("/test-triton")
async def test_triton():
    """Route de test pour vérifier la connexion avec Triton"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:8000/v2/health/ready") as response:
                if response.status == 200:
                    return {"status": "Triton est opérationnel"}
                else:
                    return {"status": "Triton ne répond pas correctement", "code": response.status}
    except Exception as e:
        return {"status": "Erreur de connexion à Triton", "error": str(e)}

@app.get("/test-triton-meta")
async def test_triton_meta():
    """Route de test pour vérifier les métadonnées du modèle Triton"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:8000/v2/models/whisper") as response:
                if response.status == 200:
                    model_meta = await response.json()
                    return {"status": "Réussi", "metadata": model_meta}
                else:
                    return {"status": "Échec", "code": response.status, "error": await response.text()}
    except Exception as e:
        return {"status": "Erreur", "error": str(e)}


# Fonction pour générer des sous-titres SRT
def generate_srt(result):
    if "chunks" in result:
        segments = result["chunks"]
    elif "segments" in result:
        segments = result["segments"]
    else:
        # Si pas de segments détaillés, créer un segment unique
        segments = [{"start": 0, "end": 100, "text": result["text"]}]

    srt_content = ""
    for i, segment in enumerate(segments):
        # Les clés peuvent varier selon la version de Whisper
        start_time = segment.get("start", 0)
        end_time = segment.get("end", start_time + 5)
        text = segment.get("text", "").strip()

        # Formater pour SRT
        srt_content += f"{i + 1}\n"
        srt_content += f"{format_time(start_time)} --> {format_time(end_time)}\n"
        srt_content += f"{text}\n\n"

    return srt_content


def format_time(seconds):
    hours = int(seconds / 3600)
    minutes = int((seconds % 3600) / 60)
    secs = seconds % 60
    milliseconds = int((secs - int(secs)) * 1000)

    return f"{hours:02d}:{minutes:02d}:{int(secs):02d},{milliseconds:03d}"


# Remplacer la fonction process_video par une version qui utilise directement Whisper
async def process_video(file_path, request_type):
    """
    Traite le fichier avec Whisper directement.
    """
    # Créer l'ID du résultat
    result_id = str(uuid.uuid4())
    result_dir = f"results/{result_id}"
    os.makedirs(result_dir, exist_ok=True)

    # Déterminer le type de fichier
    file_ext = os.path.splitext(file_path)[1].lower()

    # Métriques pour les statistiques
    start_time = time.time()

    # Si c'est une vidéo MP4, extraire l'audio en WAV
    temp_audio_path = None
    input_path = file_path

    if file_ext == '.mp4':
        temp_audio_path = f"{result_dir}/audio.wav"
        # Extraire seulement les 60 premières secondes et convertir en mono 16kHz
        os.system(f"ffmpeg -i {file_path} -t 60 -ar 16000 -ac 1 {temp_audio_path} -y")
        print(f"Audio extrait vers : {temp_audio_path}")
        input_path = temp_audio_path

    # Transcription avec Whisper
    try:
        print(f"Transcription du fichier {input_path}")
        result = whisper_model(input_path)
        print(f"Transcription réussie: {result['text'][:100]}...")

        # Récupérer le texte complet et les sous-titres
        text = result["text"]
        subtitles = generate_srt(result)

        # Préparation des fichiers résultats selon le type de requête
        result_files = {}

        if request_type == 1:  # Vidéo avec sous-titres intégrés
            if file_ext != ".mp4":
                raise HTTPException(status_code=400,
                                    detail="Seuls les fichiers MP4 sont acceptés pour ce type de traitement")

            # Enregistrer les sous-titres dans un fichier SRT
            srt_path = f"{result_dir}/subtitles.srt"
            with open(srt_path, "w", encoding="utf-8") as f:
                f.write(subtitles)

            # Utiliser ffmpeg pour incruster les sous-titres dans la vidéo
            output_path = f"{result_dir}/video_with_subtitles.mp4"
            os.system(f'ffmpeg -i {file_path} -vf subtitles={srt_path} {output_path}')

            result_files = {
                "result_url": f"/results/{result_id}/video_with_subtitles.mp4",
                "download_url": f"/download/{result_id}/video_with_subtitles.mp4"
            }

        elif request_type == 2:  # Vidéo + sous-titres séparés
            if file_ext != ".mp4":
                raise HTTPException(status_code=400,
                                    detail="Seuls les fichiers MP4 sont acceptés pour ce type de traitement")

            # Copier la vidéo originale
            video_path = f"{result_dir}/video.mp4"
            shutil.copy2(file_path, video_path)

            # Enregistrer les sous-titres dans un fichier SRT
            srt_path = f"{result_dir}/subtitles.srt"
            with open(srt_path, "w", encoding="utf-8") as f:
                f.write(subtitles)

            # Ajouter les métadonnées à la vidéo
            metadata_video_path = f"{result_dir}/video_with_metadata.mp4"
            os.system(f'ffmpeg -i {video_path} -c copy -metadata:s:s:0 language=fra {metadata_video_path}')

            result_files = {
                "video_url": f"/results/{result_id}/video_with_metadata.mp4",
                "subtitles": subtitles,
                "subtitles_url": f"/results/{result_id}/subtitles.srt",
                "download_url": f"/download/{result_id}/video_with_metadata.mp4"
            }

        elif request_type == 3:  # Sous-titres uniquement
            # Enregistrer les sous-titres dans un fichier SRT
            srt_path = f"{result_dir}/subtitles.srt"
            with open(srt_path, "w", encoding="utf-8") as f:
                f.write(subtitles)

            result_files = {
                "subtitles": subtitles,
                "download_url": f"/download/{result_id}/subtitles.srt"
            }

        elif request_type == 4:  # Texte uniquement
            # Enregistrer le texte dans un fichier TXT
            txt_path = f"{result_dir}/transcription.txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)

            result_files = {
                "text": text,
                "download_url": f"/download/{result_id}/transcription.txt"
            }

        else:
            raise HTTPException(status_code=400, detail="Type de requête non reconnu")

    except Exception as e:
        import traceback
        print(f"Erreur lors de la transcription: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Erreur lors de la transcription: {str(e)}")

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
    print("UPLOAD VIDEP")
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
    print("Process id")
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
    print("ENREGISTERE")
    try:
        # Traiter le fichier
        result, processing_time = await process_video(file_path, type_of_request)
        print("TRAITEMENT")
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

    uvicorn.run(app, host="localhost", port=8003)