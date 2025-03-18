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


async def process_video(file_path, request_type):
    """
    Traite le fichier en envoyant une requête au serveur Triton.
    """
    # Créer l'ID du résultat
    result_id = str(uuid.uuid4())
    result_dir = f"results/{result_id}"
    os.makedirs(result_dir, exist_ok=True)

    # Déterminer le type de fichier
    file_ext = os.path.splitext(file_path)[1].lower()

    # Métriques pour les statistiques
    start_time = time.time()

    # Extraire l'audio avec une compression pour réduire la taille
    audio_file_path = file_path
    if file_ext == '.mp4':
        audio_file_path = f"{result_dir}/audio.wav"
        # Convertir en mono 16kHz pour réduire la taille
        os.system(f"ffmpeg -i {file_path} -t 30 -ar 16000 -ac 1 {audio_file_path} -y")
        print(f"Audio extrait vers : {audio_file_path}")

    # Vérifier que le fichier audio existe
    if not os.path.exists(audio_file_path):
        raise Exception(f"Échec de l'extraction audio : fichier {audio_file_path} non créé")

    # Lire le fichier audio
    with open(audio_file_path, "rb") as f:
        audio_content = f.read()

    # Vérifier la taille du fichier audio
    print(f"Taille du fichier audio : {len(audio_content)} octets")

    # Encoder en base64 pour l'envoi à Triton
    audio_content_b64 = base64.b64encode(audio_content).decode('utf-8')

    # Préparer la requête pour Triton
    request_body = {
        "inputs": [
            {
                "name": "audio_file",
                "shape": [1, 1],
                "datatype": "BYTES",
                "data": [[audio_content_b64]]
            },
            {
                "name": "request_type",
                "shape": [1, 1],
                "datatype": "INT32",
                "data": [[request_type]]
            }
        ]
    }
    async with aiohttp.ClientSession() as session:
        try:
            print("6.1")
            async with session.post(
                    "http://localhost:8000/v2/models/whisper/infer",
                    json=request_body,
                    headers={"Content-Type": "application/json"}
            ) as response:
                print("6.2")
                print(response.status)
                if response.status != 200:
                    error_msg = await response.text()
                    print(error_msg)
                    raise Exception(f"Erreur de Triton: {error_msg}")
                print("6.3")
                triton_result = await response.json()
                print("6.4")
                # Extraire les résultats
                subtitles = None
                text = None

                for output in triton_result.get("outputs", []):
                    if output["name"] == "subtitles":
                        subtitles_bytes = base64.b64decode(output["data"][0])
                        subtitles = subtitles_bytes.decode('utf-8')
                    elif output["name"] == "text":
                        text_bytes = base64.b64decode(output["data"][0])
                        text = text_bytes.decode('utf-8')
        except Exception as e:
            raise Exception(f"Erreur lors de la communication avec Triton: {str(e)}")
    print("7")
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