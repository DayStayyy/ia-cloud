import json
import numpy as np
import triton_python_backend_utils as pb_utils
import base64
import torch
from transformers import pipeline
import os
import tempfile
import time
from typing import Dict, List, Tuple, Optional, Union
import warnings

warnings.filterwarnings('ignore')

# Mapping des modèles disponibles
MODELS = {
    "whisper-small": {
        "path": "openai/whisper-small",
        "supports_speaker_id": False
    },
    "whisper-medium": {
        "path": "openai/whisper-medium",
        "supports_speaker_id": False
    },
    "whisper-large": {
        "path": "openai/whisper-large-v3",
        "supports_speaker_id": False
    },
    "whisper-small-with-speaker": {
        "path": "openai/whisper-small",
        "supports_speaker_id": True,
        "speaker_model": "pyannote/speaker-diarization"
    }
}


# Classe pour gérer les modèles
class ModelManager:
    def __init__(self):
        self.models = {}
        self.speaker_diarization_model = None

    def get_model(self, model_name):
        if model_name not in self.models:
            if model_name not in MODELS:
                raise ValueError(f"Modèle {model_name} non pris en charge")

            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if device == "cuda" else torch.float32

            print(f"Chargement du modèle {model_name} sur {device}...")
            model_config = MODELS[model_name]

            self.models[model_name] = pipeline(
                "automatic-speech-recognition",
                model_config["path"],
                torch_dtype=dtype,
                device=device,
                return_timestamps=True
            )

            # Charger le modèle de diarization si nécessaire
            if model_config["supports_speaker_id"] and self.speaker_diarization_model is None:
                try:
                    from pyannote.audio import Pipeline as PyannotePipeline
                    self.speaker_diarization_model = PyannotePipeline.from_pretrained(
                        model_config["speaker_model"]
                    )
                    print("Modèle de diarization chargé avec succès")
                except Exception as e:
                    print(f"Erreur lors du chargement du modèle de diarization: {str(e)}")

        return self.models[model_name]

    def get_speaker_diarization_model(self):
        return self.speaker_diarization_model


# Singleton pour gérer les modèles
model_manager = ModelManager()


class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args['model_config'])

        # Précharger le modèle Whisper par défaut
        try:
            default_model = "whisper-small"
            self.asr_model = model_manager.get_model(default_model)
            print(f"Modèle {default_model} préchargé avec succès")
        except Exception as e:
            print(f"Erreur lors du préchargement du modèle: {str(e)}")

    def execute(self, requests):
        responses = []

        for request in requests:
            # Extraire l'audio et le type de requête
            audio_file = pb_utils.get_input_tensor_by_name(request, 'audio_file').as_numpy()[0][0]
            request_type = pb_utils.get_input_tensor_by_name(request, 'request_type').as_numpy()[0][0]

            # Paramètre optionnel : modèle à utiliser
            try:
                model_name = pb_utils.get_input_tensor_by_name(request, 'model_name')
                if model_name is not None:
                    model_name = model_name.as_numpy()[0][0].decode('utf-8')
                else:
                    model_name = "whisper-small"
            except:
                model_name = "whisper-small"

            # Paramètre optionnel : identifier les speakers
            try:
                identify_speakers = pb_utils.get_input_tensor_by_name(request, 'identify_speakers')
                if identify_speakers is not None:
                    identify_speakers = bool(identify_speakers.as_numpy()[0][0])
                else:
                    identify_speakers = False
            except:
                identify_speakers = False

            # Récupérer le modèle approprié
            if identify_speakers and not MODELS.get(model_name, {}).get("supports_speaker_id", False):
                # Utiliser un modèle qui supporte l'identification de locuteur
                for m_name, m_config in MODELS.items():
                    if m_config["supports_speaker_id"]:
                        model_name = m_name
                        break

            try:
                # Décoder l'audio depuis base64
                audio_decoded = base64.b64decode(audio_file)

                # Sauvegarder l'audio dans un fichier temporaire
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
                    temp_audio.write(audio_decoded)
                    temp_audio_path = temp_audio.name

                # Obtenir le modèle approprié
                model = model_manager.get_model(model_name)

                # Transcription avec Whisper
                transcription = model(temp_audio_path, chunk_length_s=28, return_timestamps=True)

                # Générer les sous-titres
                subtitles, text = self.process_transcription(transcription, temp_audio_path, identify_speakers)

                # Nettoyer les fichiers temporaires
                try:
                    os.unlink(temp_audio_path)
                except:
                    pass

                # Encoder les résultats
                subtitles_encoded = base64.b64encode(subtitles.encode('utf-8')).decode('utf-8')
                text_encoded = base64.b64encode(text.encode('utf-8')).decode('utf-8')

                # Créer les tenseurs de sortie
                subtitles_tensor = pb_utils.Tensor('subtitles', np.array([[subtitles_encoded]], dtype=np.object_))
                text_tensor = pb_utils.Tensor('text', np.array([[text_encoded]], dtype=np.object_))
                model_used_tensor = pb_utils.Tensor('model_used',
                                                    np.array([[model_name.encode('utf-8')]], dtype=np.object_))

                # Créer la réponse
                inference_response = pb_utils.InferenceResponse(
                    output_tensors=[subtitles_tensor, text_tensor, model_used_tensor]
                )
                responses.append(inference_response)

            except Exception as e:
                error_message = f"Erreur lors du traitement : {str(e)}"
                print(error_message)

                # Créer des tenseurs d'erreur
                error_tensor = pb_utils.Tensor('error', np.array([[error_message.encode('utf-8')]], dtype=np.object_))
                inference_response = pb_utils.InferenceResponse(
                    output_tensors=[error_tensor]
                )
                responses.append(inference_response)

        return responses

    def process_transcription(self, transcription, audio_path, identify_speakers=False):
        """
        Traite la transcription et renvoie les sous-titres et le texte.
        Si identify_speakers est activé, ajoute l'identification du locuteur.
        """
        if not identify_speakers:
            # Méthode standard - Générer les sous-titres SRT sans identification de locuteur
            subtitles = self.generate_srt(transcription)
            text = transcription["text"]
        else:
            # Méthode avec identification de locuteur
            try:
                speaker_diarization_model = model_manager.get_speaker_diarization_model()
                if speaker_diarization_model is None:
                    raise ValueError("Modèle de diarization non disponible")

                # Effectuer la diarization
                diarization = speaker_diarization_model(audio_path)

                # Fusionner la transcription et la diarization
                subtitles, text = self.merge_transcription_with_speakers(transcription, diarization)
            except Exception as e:
                print(f"Erreur lors de l'identification des locuteurs: {str(e)}")
                # Fallback à la méthode standard
                subtitles = self.generate_srt(transcription)
                text = transcription["text"]

        return subtitles, text

    def merge_transcription_with_speakers(self, transcription, diarization):
        """
        Fusionne la transcription et la diarization pour créer des sous-titres avec identification des locuteurs
        """
        # Structure pour stocker les segments avec identification de locuteur
        segments_with_speakers = []

        # Extraire les segments de transcription
        if "chunks" in transcription:
            segments = transcription["chunks"]
        elif "segments" in transcription:
            segments = transcription["segments"]
        else:
            segments = []

        # Pour chaque segment de transcription, trouver le locuteur correspondant
        for segment in segments:
            start_time = segment["timestamp"][0] if "timestamp" in segment else segment.get("start", 0)
            end_time = segment["timestamp"][1] if "timestamp" in segment else segment.get("end", start_time + 5)

            # Trouver le locuteur principal pour ce segment
            speaker = self.find_main_speaker(diarization, start_time, end_time)

            segments_with_speakers.append({
                "start": start_time,
                "end": end_time,
                "text": segment.get("text", "").strip(),
                "speaker": speaker
            })

        # Générer les sous-titres SRT avec identification de locuteur
        srt_content = ""
        full_text = ""

        for i, segment in enumerate(segments_with_speakers):
            # Format SRT
            srt_content += f"{i + 1}\n"
            srt_content += f"{self.format_time(segment['start'])} --> {self.format_time(segment['end'])}\n"
            srt_content += f"[{segment['speaker']}] {segment['text']}\n\n"

            # Texte complet
            full_text += f"[{segment['speaker']}] {segment['text']} "

        return srt_content, full_text.strip()

    def find_main_speaker(self, diarization, start_time, end_time):
        """
        Trouve le locuteur principal dans un intervalle de temps donné
        """
        # Mapping des locuteurs détectés dans l'intervalle
        speakers_time = {}

        # Parcourir les segments de diarization pour l'intervalle
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            # Si le segment se chevauche avec l'intervalle
            if turn.start < end_time and turn.end > start_time:
                # Calculer la durée du chevauchement
                overlap_start = max(turn.start, start_time)
                overlap_end = min(turn.end, end_time)
                overlap_duration = overlap_end - overlap_start

                # Ajouter au compteur de temps pour ce locuteur
                if speaker not in speakers_time:
                    speakers_time[speaker] = 0
                speakers_time[speaker] += overlap_duration

        # Trouver le locuteur avec le plus de temps de parole
        if speakers_time:
            main_speaker = max(speakers_time.items(), key=lambda x: x[1])[0]
            return main_speaker
        else:
            return "Inconnu"

    def generate_srt(self, result):
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
            srt_content += f"{self.format_time(start)} --> {self.format_time(end)}\n"
            srt_content += f"{text}\n\n"

        return srt_content

    def format_time(self, seconds):
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