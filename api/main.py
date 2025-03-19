import base64
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import Optional, List
import shutil
import os
import time
import uuid
import json
from datetime import datetime
from pydantic import BaseModel
import aiohttp
import ffmpeg


# Définition des modèles Pydantic
class ModelInfo(BaseModel):
    name: str
    description: str
    supports_speaker_id: bool


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

# Liste des modèles disponibles
AVAILABLE_MODELS = [
    ModelInfo(
        name="whisper-small",
        description="Modèle Whisper Small - Rapide et efficace",
        supports_speaker_id=False
    ),
    ModelInfo(
        name="whisper-medium",
        description="Modèle Whisper Medium - Bon équilibre entre vitesse et précision",
        supports_speaker_id=False
    ),
    ModelInfo(
        name="whisper-large",
        description="Modèle Whisper Large - Haute précision (plus lent)",
        supports_speaker_id=False
    ),
    ModelInfo(
        name="whisper-small-with-speaker",
        description="Modèle Whisper Small avec identification des locuteurs",
        supports_speaker_id=True
    )
]

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
        "inferences_by_model": {},
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
    # Initialiser les statistiques pour chaque modèle
    for model in AVAILABLE_MODELS:
        initial_stats["inferences_by_model"][model.name] = 0

    with open(STATS_FILE, "w") as f:
        json.dump(initial_stats, f, indent=4)


# Fonction pour charger les statistiques
def load_statistics():
    try:
        with open(STATS_FILE, "r") as f:
            stats = json.load(f)

            # S'assurer que tous les modèles sont présents dans les statistiques
            if "inferences_by_model" not in stats:
                stats["inferences_by_model"] = {}

            for model in AVAILABLE_MODELS:
                if model.name not in stats["inferences_by_model"]:
                    stats["inferences_by_model"][model.name] = 0

            return stats
    except (FileNotFoundError, json.JSONDecodeError):
        # En cas d'erreur, retourner les statistiques initiales
        return {
            "avg_processing_time": 0,
            "total_inferences": 0,
            "inferences_by_type": {"1": 0, "2": 0, "3": 0, "4": 0},
            "inferences_by_model": {model.name: 0 for model in AVAILABLE_MODELS},
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
def update_statistics(request_type, processing_time, model_name="whisper-small", success=True):
    stats = load_statistics()

    # Réinitialisation temporaire si nécessaire
    if "processing_times" not in stats or len(stats["processing_times"]) > 1000:
        stats["processing_times"] = []

    request_type_str = str(request_type)

    if success:
        # Augmenter le nombre total d'inférences
        stats["total_inferences"] += 1
        stats["inferences_by_type"][request_type_str] += 1

        # Augmenter le nombre d'inférences pour ce modèle
        if model_name in stats["inferences_by_model"]:
            stats["inferences_by_model"][model_name] += 1

        # Mettre à jour le temps de traitement moyen
        if processing_time > 0 and processing_time < 3600:
            stats["processing_times"].append(processing_time)

        stats["avg_processing_time"] = sum(stats["processing_times"]) / len(stats["processing_times"]) if stats[
            "processing_times"] else 0

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
            async with session.get("http://triton:8000/v2/health/ready") as response:
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
            async with session.get("http://triton:8000/v2/models/whisper") as response:
                if response.status == 200:
                    model_meta = await response.json()
                    return {"status": "Réussi", "metadata": model_meta}
                else:
                    return {"status": "Échec", "code": response.status, "error": await response.text()}
    except Exception as e:
        return {"status": "Erreur", "error": str(e)}


# Fonction pour générer des sous-titres SRT
def format_time(seconds):
    """
    Formate un temps en secondes au format SRT (HH:MM:SS,MMM).
    """
    hours = int(seconds / 3600)
    seconds %= 3600
    minutes = int(seconds / 60)
    seconds %= 60
    # Extraction précise des millisecondes
    milliseconds = round((seconds - int(seconds)) * 1000)
    seconds = int(seconds)

    # Format exact avec une virgule (pas un point) entre secondes et millisecondes
    formatted_time = f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

    return formatted_time


def generate_srt(result):
    """
    Génère un contenu SRT à partir du résultat de Whisper.
    """
    srt_content = ""
    offset = 0

    # Déterminer la source des segments (chunks ou segments)
    if "chunks" in result:
        segments = result["chunks"]
    elif "segments" in result:
        segments = result["segments"]
    else:
        # Si pas de segments détaillés, créer un segment unique
        return f"1\n00:00:00,000 --> 00:01:00,000\n{result['text'].strip()}\n\n"

    for index, segment in enumerate(segments):
        # Vérifier le format du timestamp (peut être ["timestamp"][0/1] ou start/end)
        if "timestamp" in segment and isinstance(segment["timestamp"], list) and len(segment["timestamp"]) == 2:
            start = offset + segment["timestamp"][0]
            end = offset + segment["timestamp"][1]

            # Vérifier si start > end (indication d'un nouveau segment)
            if start > end:
                offset += 28  # Même valeur que chunk_length_s utilisée dans le traitement
                continue
        else:
            # Format alternatif avec start/end
            start = segment.get("start", 0)
            end = segment.get("end", start + 5)  # Durée par défaut de 5 secondes

        # Récupérer le texte
        text = segment.get("text", "").strip()

        # Formater pour SRT
        srt_content += f"{index + 1}\n"
        srt_content += f"{format_time(start)} --> {format_time(end)}\n"
        srt_content += f"{text}\n\n"

    return srt_content


async def process_video(file_path, request_type, model_name="whisper-small", identify_speakers=False):
    """
    Traite le fichier en envoyant une requête au serveur Triton.
    """
    # Créer l'ID du résultat
    result_id = str(uuid.uuid4())
    result_dir = f"results/{result_id}"
    os.makedirs(result_dir, exist_ok=True)

    # Déterminer le type de fichier
    file_ext = os.path.splitext(file_path)[1].lower()
    file_name = os.path.basename(file_path).replace(file_ext, "")  # Nom du fichier sans extension

    # Métriques pour les statistiques
    start_time = time.time()

    # Si c'est une vidéo MP4, extraire l'audio en WAV
    temp_audio_path = None
    input_path = file_path

    if file_ext == '.mp4':
        temp_audio_path = f"{result_dir}/audio-{file_name}.wav"
        try:
            stream = ffmpeg.input(file_path)
            stream = ffmpeg.output(stream, temp_audio_path)  # Pas de paramètres ar ou ac
            ffmpeg.run(stream, overwrite_output=True)
            print(f"Audio extrait vers : {temp_audio_path}")
            input_path = temp_audio_path
        except ffmpeg.Error as e:
            print(f"Erreur lors de l'extraction audio: {e.stderr.decode() if e.stderr else str(e)}")
            raise HTTPException(status_code=500, detail="Erreur lors de l'extraction audio")

    try:
        # Lire le fichier audio
        with open(input_path, "rb") as f:
            audio_content = f.read()

        # Vérifier la taille du fichier audio
        print(f"Taille du fichier audio : {len(audio_content)} octets")

        # Encoder en base64 pour l'envoi à Triton
        audio_content_b64 = base64.b64encode(audio_content).decode('utf-8')

        # Préparer la requête pour Triton avec les paramètres supplémentaires
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
                },
                {
                    "name": "model_name",
                    "shape": [1, 1],
                    "datatype": "BYTES",
                    "data": [[model_name.encode('utf-8')]]
                },
                {
                    "name": "identify_speakers",
                    "shape": [1, 1],
                    "datatype": "BOOL",
                    "data": [[identify_speakers]]
                }
            ]
        }

        # Envoyer la requête à Triton
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                        "http://triton:8000/v2/models/whisper/infer",
                        json=request_body,
                        headers={"Content-Type": "application/json"}
                ) as response:
                    if response.status != 200:
                        error_msg = await response.text()
                        print(error_msg)
                        raise Exception(f"Erreur de Triton: {error_msg}")

                    triton_result = await response.json()

                    # Extraire les résultats
                    subtitles = None
                    text = None
                    model_used = None
                    error = None

                    for output in triton_result.get("outputs", []):
                        if output["name"] == "subtitles":
                            subtitles_bytes = base64.b64decode(output["data"][0])
                            subtitles = subtitles_bytes.decode('utf-8')
                        elif output["name"] == "text":
                            text_bytes = base64.b64decode(output["data"][0])
                            text = text_bytes.decode('utf-8')
                        elif output["name"] == "model_used":
                            model_used_bytes = base64.b64decode(output["data"][0])
                            model_used = model_used_bytes.decode('utf-8')
                        elif output["name"] == "error":
                            error_bytes = base64.b64decode(output["data"][0])
                            error = error_bytes.decode('utf-8')
                            raise Exception(f"Erreur dans le traitement du modèle: {error}")
            except Exception as e:
                raise Exception(f"Erreur lors de la communication avec Triton: {str(e)}")

        # Préparation des fichiers résultats selon le type de requête
        result_files = {}

        if request_type == 1:  # Vidéo avec sous-titres intégrés
            if file_ext != ".mp4":
                raise HTTPException(status_code=400,
                                    detail="Seuls les fichiers MP4 sont acceptés pour ce type de traitement")

            # Enregistrer les sous-titres dans un fichier SRT
            subtitle_file = f"{result_dir}/sub-{file_name}.en.srt"
            with open(subtitle_file, "w", encoding="utf-8") as f:
                f.write(subtitles)

            # Utiliser ffmpeg pour incruster les sous-titres dans la vidéo
            output_path = f"{result_dir}/video_with_subtitles.mp4"

            try:
                video_input_stream = ffmpeg.input(file_path)
                stream = ffmpeg.output(video_input_stream, output_path,
                                       vf=f"subtitles={subtitle_file}")
                ffmpeg.run(stream, overwrite_output=True)

                result_files = {
                    "result_url": f"/results/{result_id}/video_with_subtitles.mp4",
                    "download_url": f"/download/{result_id}/video_with_subtitles.mp4",
                    "model_used": model_used,
                    "speaker_identification": identify_speakers
                }
            except ffmpeg.Error as e:
                print(f"Erreur FFmpeg: {e.stderr.decode() if e.stderr else str(e)}")
                raise HTTPException(status_code=500, detail=f"Erreur lors de l'incrustation des sous-titres: {str(e)}")

        elif request_type == 2:  # Vidéo + sous-titres séparés
            if file_ext != ".mp4":
                raise HTTPException(status_code=400,
                                    detail="Seuls les fichiers MP4 sont acceptés pour ce type de traitement")

            # Enregistrer les sous-titres dans un fichier SRT
            subtitle_file = f"{result_dir}/sub-{file_name}.en.srt"
            with open(subtitle_file, "w", encoding="utf-8") as f:
                f.write(subtitles)

            # Copier la vidéo originale
            video_path = f"{result_dir}/video.mp4"
            shutil.copy2(file_path, video_path)

            # Ajouter les métadonnées à la vidéo
            output_path = f"{result_dir}/video_with_metadata.mp4"

            try:
                video_input_stream = ffmpeg.input(file_path)
                subtitle_input_stream = ffmpeg.input(subtitle_file)

                stream = ffmpeg.output(
                    video_input_stream, subtitle_input_stream, output_path,
                    **{"c": "copy", "c:s": "mov_text"},
                    **{"metadata:s:s:0": "language=fra",
                       "metadata:s:s:1": f"title=sub-{file_name}"}
                )
                ffmpeg.run(stream, overwrite_output=True)

                result_files = {
                    "video_url": f"/results/{result_id}/video_with_metadata.mp4",
                    "subtitles": subtitles,
                    "subtitles_url": f"/results/{result_id}/sub-{file_name}.en.srt",
                    "download_url": f"/download/{result_id}/video_with_metadata.mp4",
                    "model_used": model_used,
                    "speaker_identification": identify_speakers
                }
            except ffmpeg.Error as e:
                print(f"Erreur FFmpeg: {e.stderr.decode() if e.stderr else str(e)}")
                raise HTTPException(status_code=500, detail="Erreur lors de l'ajout des métadonnées")

        elif request_type == 3:  # Sous-titres uniquement
            # Enregistrer les sous-titres dans un fichier SRT
            subtitle_file = f"{result_dir}/sub-{file_name}.en.srt"
            with open(subtitle_file, "w", encoding="utf-8") as f:
                f.write(subtitles)

            result_files = {
                "subtitles": subtitles,
                "download_url": f"/download/{result_id}/sub-{file_name}.en.srt",
                "model_used": model_used,
                "speaker_identification": identify_speakers
            }

        elif request_type == 4:  # Texte uniquement
            # Enregistrer le texte dans un fichier TXT
            txt_path = f"{result_dir}/transcription.txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)

            result_files = {
                "text": text,
                "download_url": f"/download/{result_id}/transcription.txt",
                "model_used": model_used,
                "speaker_identification": identify_speakers
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

    return result_files, processing_time, model_used


@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/stats")
async def stats_page(request: Request):
    """
    Route pour afficher la page des statistiques
    """
    return templates.TemplateResponse("stats.html", {"request": request})


@app.get("/models")
async def models_page(request: Request):
    """
    Route pour afficher la page de gestion des modèles
    """
    return templates.TemplateResponse("models.html", {"request": request})


@app.get("/api/models")
async def get_available_models():
    """
    API pour récupérer la liste des modèles disponibles
    """
    return {
        "models": [model.dict() for model in AVAILABLE_MODELS]
    }


# Route pour l'upload de vidéo
@app.post("/upload-video")
async def upload_video(
        video: UploadFile = File(...),
        type_of_request: int = Form(...),
        model_name: str = Form("whisper-small"),
        identify_speakers: bool = Form(False),
):
    print(f"UPLOAD VIDEO - Modèle: {model_name}, Speaker ID: {identify_speakers}")
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

    # Vérifier que le modèle existe
    model_exists = False
    for available_model in AVAILABLE_MODELS:
        if available_model.name == model_name:
            model_exists = True
            # Si l'identification des locuteurs est demandée, vérifier que le modèle la supporte
            if identify_speakers and not available_model.supports_speaker_id:
                # Trouver un modèle qui supporte l'identification des locuteurs
                for m in AVAILABLE_MODELS:
                    if m.supports_speaker_id:
                        model_name = m.name
                        break
                else:
                    raise HTTPException(
                        status_code=400,
                        detail="Aucun modèle disponible ne supporte l'identification des locuteurs"
                    )
            break

    if not model_exists:
        raise HTTPException(
            status_code=400,
            detail=f"Le modèle {model_name} n'existe pas"
        )

    # Mettre à jour les statistiques - ajout d'un processus en attente
    stats = load_statistics()
    stats["pending_processes"] += 1
    stats["pending_by_type"][str(type_of_request)] += 1
    save_statistics(stats)

    # Créer un ID unique pour ce traitement
    processing_id = str(uuid.uuid4())
    print(f"Process ID: {processing_id}")

    # Créer le chemin du fichier temporaire
    file_path = f"uploads/{processing_id}{file_ext}"

    # Enregistrer le fichier uploadé
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
    except Exception as e:
        # En cas d'erreur, mettre à jour les statistiques
        update_statistics(type_of_request, 0, model_name, success=False)
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'enregistrement du fichier: {str(e)}")

    print("Fichier enregistré, début du traitement...")

    try:
        # Traiter le fichier avec les paramètres spécifiés
        result, processing_time, model_used = await process_video(
            file_path,
            type_of_request,
            model_name,
            identify_speakers
        )

        print(f"Traitement terminé en {processing_time:.2f}s avec le modèle {model_used}")

        # Mettre à jour les statistiques - traitement réussi
        update_statistics(type_of_request, processing_time, model_used, success=True)

        return JSONResponse(content=result)

    except Exception as e:
        # En cas d'erreur, mettre à jour les statistiques
        update_statistics(type_of_request, 0, model_name, success=False)
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
        "inferences_by_model": stats["inferences_by_model"],
        "pending_processes": stats["pending_processes"],
        "pending_by_type": stats["pending_by_type"],
        "failed_processes": stats["failed_processes"],
        "failed_by_type": stats["failed_by_type"]
    }


# Route pour obtenir les statistiques de Triton
@app.get("/api/triton-statistics")
async def get_triton_statistics():
    """Récupère les statistiques de Triton Inference Server"""
    try:
        async with aiohttp.ClientSession() as session:
            # Statistiques générales du serveur
            async with session.get("http://triton:8000/v2/metrics") as response:
                if response.status == 200:
                    metrics_text = await response.text()

                    # Extraction simple de quelques métriques clés
                    inference_count = 0
                    model_load_time = 0
                    inference_exec_time = 0

                    # Analyse très basique des métriques Prometheus
                    for line in metrics_text.split('\n'):
                        if 'nv_inference_count' in line and not line.startswith('#'):
                            parts = line.split()
                            if len(parts) >= 2:
                                try:
                                    inference_count = int(float(parts[1]))
                                except:
                                    pass
                        elif 'nv_inference_exec_count' in line and not line.startswith('#'):
                            parts = line.split()
                            if len(parts) >= 2:
                                try:
                                    inference_exec_time = float(parts[1])
                                except:
                                    pass

                    return {
                        "status": "success",
                        "inference_count": inference_count,
                        "inference_exec_time": inference_exec_time,
                    }
                else:
                    return {"status": "error",
                            "message": f"Erreur {response.status} lors de la récupération des métriques"}
    except Exception as e:
        return {"status": "error", "message": f"Erreur de connexion: {str(e)}"}


# Point d'entrée pour uvicorn
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8003)