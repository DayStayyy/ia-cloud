import triton_python_backend_utils as pb_utils
import numpy as np
import torch
import tempfile
import os
import json
import base64
from transformers import pipeline


class TritonPythonModel:
    def initialize(self, args):
        self.model_config = pb_utils.get_model_config(args['model_config'])

        # Initialiser le modèle Whisper sur GPU si disponible
        device = 0 if torch.cuda.is_available() else -1
        self.whisper = pipeline("automatic-speech-recognition",
                                "openai/whisper-medium.en",
                                torch_dtype=torch.float16,
                                device=device,
                                return_timestamps=True)

        # Configuration pour les fichiers temporaires
        self.temp_dir = tempfile.mkdtemp()
        print(f"Initialized Whisper model on device: {device}")

    def execute(self, requests):
        responses = []

        for request in requests:
            # Extraire les données d'entrée
            audio_file = pb_utils.get_input_tensor_by_name(request, "audio_file")
            audio_data = audio_file.as_numpy()[0]

            request_type_tensor = pb_utils.get_input_tensor_by_name(request, "request_type")
            request_type = request_type_tensor.as_numpy()[0][0]

            # Déterminer le type de fichier et l'extension
            is_mp4 = False
            if audio_data[:4] == b'\x00\x00\x00\x18' or audio_data[:4] == b'\x00\x00\x00\x20':
                file_ext = ".mp4"
                is_mp4 = True
            else:
                file_ext = ".wav"

            # Sauvegarder les données audio dans un fichier temporaire
            temp_file = os.path.join(self.temp_dir, f"temp_{id(request)}{file_ext}")
            with open(temp_file, "wb") as f:
                f.write(audio_data)

            try:
                # Si c'est une vidéo, extraire l'audio
                if is_mp4:
                    audio_file = os.path.join(self.temp_dir, f"audio_{id(request)}.wav")
                    os.system(f"ffmpeg -i {temp_file} -q:a 0 -map a {audio_file} -y")
                    temp_file = audio_file

                # Transcription avec Whisper
                result = self.whisper(temp_file)

                # Extraire le texte
                text = result["text"]

                # Générer des sous-titres SRT
                subtitles = self._generate_srt(result)

                # Créer les tenseurs de sortie
                subtitles_tensor = pb_utils.Tensor("subtitles",
                                                   np.array([subtitles.encode('utf-8')], dtype=np.object_))
                text_tensor = pb_utils.Tensor("text",
                                              np.array([text.encode('utf-8')], dtype=np.object_))

                # Créer la réponse
                inference_response = pb_utils.InferenceResponse(
                    output_tensors=[subtitles_tensor, text_tensor])
                responses.append(inference_response)

            except Exception as e:
                error = pb_utils.TritonError(f"Error processing request: {str(e)}")
                responses.append(pb_utils.InferenceResponse(error=error))

            finally:
                # Nettoyer les fichiers temporaires
                if os.path.exists(temp_file):
                    os.remove(temp_file)

        return responses

    def _generate_srt(self, result):
        """Génère un fichier SRT à partir des résultats de Whisper"""
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
            srt_content += f"{self._format_time(start_time)} --> {self._format_time(end_time)}\n"
            srt_content += f"{text}\n\n"

        return srt_content

    def _format_time(self, seconds):
        """Convertit les secondes en format SRT (HH:MM:SS,mmm)"""
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)
        secs = seconds % 60
        milliseconds = int((secs - int(secs)) * 1000)

        return f"{hours:02d}:{minutes:02d}:{int(secs):02d},{milliseconds:03d}"

    def finalize(self):
        # Nettoyer le répertoire temporaire
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)