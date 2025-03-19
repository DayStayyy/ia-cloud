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
import ffmpeg

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

def refresh_statistics():
    stats = load_statistics()
    stats["avg_processing_time"] = sum(stats["processing_times"]) / len(stats["processing_times"])
    save_statistics(stats)


# Fonction pour mettre à jour les statistiques après un traitement
def update_statistics(request_type, processing_time, success=True):
    stats = load_statistics()
    stats["processing_times"] = []  # Réinitialisation temporaire

    request_type_str = str(request_type)

    if success:
        # Augmenter le nombre total d'inférences
        stats["total_inferences"] += 1
        stats["inferences_by_type"][request_type_str] += 1

        # Mettre à jour le temps de traitement moyen
        stats["processing_times"].append(processing_time)
        if processing_time > 0 and processing_time < 3600:
            stats["processing_times"].append(processing_time)
        processing_time_seconds = processing_time / 1000  # exemple
        stats["processing_times"].append(processing_time_seconds)

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
def format_time(seconds):
    """
    Formate un temps en secondes au format SRT (HH:MM:SS,MMM).
    Similaire à la fonction utilisée dans test_pipeline_3.py
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
    Implémentation améliorée basée sur test_pipeline_3.py
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


# Remplacer la fonction process_video par une version qui utilise directement Whisper


async def process_video(file_path, request_type):
    """
    Traite le fichier avec Whisper directement.
    Reproduit exactement le comportement de test_pipeline_3.py
    """
    import math  # Nécessaire pour la fonction format_time identique à test_pipeline_3.py

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
            # Utiliser exactement la même méthode que test_pipeline_3.py
            stream = ffmpeg.input(file_path)
            stream = ffmpeg.output(stream, temp_audio_path)  # Pas de paramètres ar ou ac
            ffmpeg.run(stream, overwrite_output=True)
            print(f"Audio extrait vers : {temp_audio_path}")
            input_path = temp_audio_path
        except ffmpeg.Error as e:
            print(f"Erreur lors de l'extraction audio: {e.stderr.decode() if e.stderr else str(e)}")
            raise HTTPException(status_code=500, detail="Erreur lors de l'extraction audio")

    # Transcription avec Whisper - utiliser exactement les mêmes paramètres
    try:
        print(f"Transcription du fichier {input_path}")
        # En fonction du device, adapter le dtype comme dans test_pipeline_3.py
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        # Utiliser le modèle identique à test_pipeline_3.py
        transcription = whisper_model(input_path, chunk_length_s=28, return_timestamps=True)
        print(f"Transcription réussie: {transcription['text'][:100]}...")

        # Utiliser la même méthode de génération de sous-titres que test_pipeline_3.py
        # Fonction format_time identique à test_pipeline_3.py
        def format_time(seconds):
            hours = math.floor(seconds / 3600)
            seconds %= 3600
            minutes = math.floor(seconds / 60)
            seconds %= 60
            milliseconds = round((seconds - math.floor(seconds)) * 1000)
            seconds = math.floor(seconds)
            formatted_time = f"{hours:02d}:{minutes:02d}:{seconds:01d},{milliseconds:03d}"
            return formatted_time

        # Préparation des fichiers résultats selon le type de requête
        result_files = {}

        if request_type == 1:  # Vidéo avec sous-titres intégrés
            if file_ext != ".mp4":
                raise HTTPException(status_code=400,
                                    detail="Seuls les fichiers MP4 sont acceptés pour ce type de traitement")

            # Générer les sous-titres avec EXACTEMENT la même méthode que test_pipeline_3.py
            subtitle_file = f"{result_dir}/sub-{file_name}.en.srt"
            text = ""
            offset = 0

            for index, chunk in enumerate(transcription["chunks"]):
                start = offset + chunk["timestamp"][0]
                end = offset + chunk["timestamp"][1]
                if start > end:
                    # chunk is the delimitation of a new segment
                    offset += 28
                    continue
                text_chunk = chunk["text"]
                segment_start = format_time(start)
                segment_end = format_time(end)
                text += f"{str(index + 1)}\n"
                text += f"{segment_start} --> {segment_end}\n"
                text += f"{text_chunk}\n"
                text += "\n"

            # Écrire exactement comme dans test_pipeline_3.py
            with open(subtitle_file, "w") as f:
                f.write(text)

            # Utiliser ffmpeg exactement comme dans test_pipeline_3.py
            output_path = f"{result_dir}/video_with_subtitles.mp4"

            try:
                # Reproduire EXACTEMENT la même commande ffmpeg
                video_input_stream = ffmpeg.input(file_path)
                # IMPORTANT: Pas de subtitle_input_stream car on utilise embedded_subtitle=True
                # comme dans test_pipeline_3.py

                # Méthode avec sous-titres incrustés
                stream = ffmpeg.output(video_input_stream, output_path,
                                       vf=f"subtitles={subtitle_file}")
                ffmpeg.run(stream, overwrite_output=True)

                result_files = {
                    "result_url": f"/results/{result_id}/video_with_subtitles.mp4",
                    "download_url": f"/download/{result_id}/video_with_subtitles.mp4"
                }
            except ffmpeg.Error as e:
                print(f"Erreur FFmpeg: {e.stderr.decode() if e.stderr else str(e)}")
                # Ne pas utiliser de méthode alternative pour rester fidèle à test_pipeline_3.py
                raise HTTPException(status_code=500, detail=f"Erreur lors de l'incrustation des sous-titres: {str(e)}")

        # Le reste du code pour les autres types de requêtes...
        elif request_type == 2:  # Vidéo + sous-titres séparés
            if file_ext != ".mp4":
                raise HTTPException(status_code=400,
                                    detail="Seuls les fichiers MP4 sont acceptés pour ce type de traitement")

            # Utiliser la même méthode de génération que pour request_type 1
            subtitle_file = f"{result_dir}/sub-{file_name}.en.srt"
            text = ""
            offset = 0

            for index, chunk in enumerate(transcription["chunks"]):
                start = offset + chunk["timestamp"][0]
                end = offset + chunk["timestamp"][1]
                if start > end:
                    offset += 28
                    continue
                text_chunk = chunk["text"]
                segment_start = format_time(start)
                segment_end = format_time(end)
                text += f"{str(index + 1)}\n"
                text += f"{segment_start} --> {segment_end}\n"
                text += f"{text_chunk}\n"
                text += "\n"

            with open(subtitle_file, "w") as f:
                f.write(text)

            # Copier la vidéo originale
            video_path = f"{result_dir}/video.mp4"
            shutil.copy2(file_path, video_path)

            # Utiliser la méthode de test_pipeline_3.py pour les sous-titres séparés (embedded_subtitle=False)
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
                    "subtitles": text,
                    "subtitles_url": f"/results/{result_id}/sub-{file_name}.en.srt",
                    "download_url": f"/download/{result_id}/video_with_metadata.mp4"
                }
            except ffmpeg.Error as e:
                print(f"Erreur FFmpeg: {e.stderr.decode() if e.stderr else str(e)}")
                raise HTTPException(status_code=500, detail="Erreur lors de l'ajout des métadonnées")

        elif request_type == 3:  # Sous-titres uniquement
            # Utiliser la même méthode de génération
            subtitle_file = f"{result_dir}/sub-{file_name}.en.srt"
            text = ""
            offset = 0

            for index, chunk in enumerate(transcription["chunks"]):
                start = offset + chunk["timestamp"][0]
                end = offset + chunk["timestamp"][1]
                if start > end:
                    offset += 28
                    continue
                text_chunk = chunk["text"]
                segment_start = format_time(start)
                segment_end = format_time(end)
                text += f"{str(index + 1)}\n"
                text += f"{segment_start} --> {segment_end}\n"
                text += f"{text_chunk}\n"
                text += "\n"

            with open(subtitle_file, "w") as f:
                f.write(text)

            result_files = {
                "subtitles": text,
                "download_url": f"/download/{result_id}/sub-{file_name}.en.srt"
            }

        elif request_type == 4:  # Texte uniquement
            # Enregistrer le texte dans un fichier TXT
            txt_path = f"{result_dir}/transcription.txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(transcription["text"])

            result_files = {
                "text": transcription["text"],
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


def test_srt_generation(file_path):
    """
    Fonction de test pour générer un SRT et vérifier sa validité.
    Version avec gestion améliorée des chemins de fichiers.
    """
    import os
    import torch
    from transformers import pipeline
    import ffmpeg
    import subprocess
    import shlex

    # Créer un dossier temporaire sans espaces ni caractères spéciaux
    temp_dir = "temp_srt_test"
    os.makedirs(temp_dir, exist_ok=True)

    # Extraire l'audio si c'est une vidéo
    file_ext = os.path.splitext(file_path)[1].lower()
    if file_ext == '.mp4':
        audio_path = os.path.join(temp_dir, "audio.wav")
        try:
            stream = ffmpeg.input(file_path)
            stream = ffmpeg.output(stream, audio_path, ar=16000, ac=1)
            ffmpeg.run(stream, overwrite_output=True)
            print(f"Audio extrait vers : {audio_path}")
            input_path = audio_path
        except Exception as e:
            print(f"Erreur lors de l'extraction audio: {str(e)}")
            return None
    else:
        input_path = file_path

    # Initialiser Whisper
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Utilisation de l'appareil: {device}")
    whisper_model = pipeline("automatic-speech-recognition",
                             "openai/whisper-small.en",
                             torch_dtype=torch.float32,
                             device=device,
                             return_timestamps=True)

    # Transcription
    print(f"Transcription du fichier {input_path}")
    result = whisper_model(input_path, chunk_length_s=28, return_timestamps=True)
    print(f"Transcription réussie. Texte complet: {result['text'][:100]}...")

    # Générer le SRT avec notre fonction
    srt_content = generate_srt(result)

    # Enregistrer dans un fichier pour test (dans le dossier temporaire)
    srt_path = os.path.join(temp_dir, "subtitles.srt")
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write(srt_content)
    print(f"\nFichier SRT enregistré: {os.path.abspath(srt_path)}")

    # Test d'incrustation - Méthode 1 (avec ffmpeg-python)
    output_path = os.path.join(temp_dir, "output_with_subs.mp4")
    try:
        print("\nTentative avec ffmpeg-python:")
        video_input = ffmpeg.input(file_path)
        srt_abs_path = os.path.abspath(srt_path).replace('\\', '/')  # Normaliser les chemins pour Windows

        # Option 1: filtres séparés
        stream = (
            video_input
            .filter('subtitles', filename=srt_abs_path, force_style='FontSize=24,Alignment=2')
            .output(output_path, acodec='copy')
        )

        ffmpeg.run(stream, overwrite_output=True)

        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            print(f"Vidéo générée avec succès (méthode 1): {os.path.abspath(output_path)}")
            return srt_path

    except ffmpeg.Error as e:
        print(f"Erreur avec ffmpeg-python: {e.stderr.decode() if e.stderr else str(e)}")

    # Méthode 2: Utiliser subprocess avec des arguments correctement échappés
    try:
        print("\nTentative avec subprocess:")
        output_path2 = os.path.join(temp_dir, "output_with_subs2.mp4")

        # Échapper correctement le chemin du fichier SRT pour FFmpeg
        # Sur Windows, utilisez des guillemets doubles et échappez les guillemets internes
        if os.name == 'nt':  # Windows
            srt_arg = f"subtitles={srt_abs_path}:force_style='FontSize=24,Alignment=2'"
        else:  # Linux/Mac
            srt_arg = f"subtitles='{srt_abs_path}':force_style='FontSize=24,Alignment=2'"

        cmd = [
            'ffmpeg',
            '-i', file_path,
            '-vf', srt_arg,
            '-c:a', 'copy',
            '-y',  # Forcer l'écrasement
            output_path2
        ]

        print(f"Commande: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"Erreur subprocess: {result.stderr}")

            # Méthode 3: Dernière tentative simplifiée
            print("\nDernière tentative simplifiée:")
            output_path3 = os.path.join(temp_dir, "output_with_subs3.mp4")

            # Utilisez une approche très simple sans styles
            cmd_simple = [
                'ffmpeg',
                '-i', file_path,
                '-vf', f"subtitles={srt_abs_path}",
                '-c:a', 'copy',
                '-y',
                output_path3
            ]

            print(f"Commande simplifiée: {' '.join(cmd_simple)}")
            result = subprocess.run(cmd_simple, capture_output=True, text=True)

            if result.returncode == 0 and os.path.exists(output_path3) and os.path.getsize(output_path3) > 0:
                print(f"Vidéo générée avec succès (méthode 3): {os.path.abspath(output_path3)}")
                return srt_path
            else:
                print(f"Échec de toutes les méthodes. Erreur: {result.stderr}")
        else:
            print(f"Vidéo générée avec succès (méthode 2): {os.path.abspath(output_path2)}")
            return srt_path

    except Exception as e:
        print(f"Erreur générale: {str(e)}")

    return srt_path

@app.get("/stats")
async def stats_page(request: Request):
    """
    Route pour afficher la page des statistiques
    """
    refresh_statistics()
    return templates.TemplateResponse("stats.html", {"request": request})

# Exemple d'utilisation:
# test_srt_generation("chemin/vers/video.mp4")

# Exemple d'utilisation:
# test_srt_generation("chemin/vers/video.mp4")


# Point d'entrée pour uvicorn
if __name__ == "__main__":
    #test_srt_generation("../input.mp4")
    import uvicorn

    uvicorn.run(app, host="localhost", port=8003)