# Image NVIDIA Triton de base
FROM nvcr.io/nvidia/tritonserver:23.10-py3

# Installation des dépendances système
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    ffmpeg \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Définir le répertoire de travail
WORKDIR /opt/tritonserver

# Créer le répertoire des modèles
RUN mkdir -p /models/whisper/1

# Copier les fichiers de configuration et modèle
COPY model_repository/whisper/config.pbtxt /models/whisper/
COPY model_repository/whisper/model.py /models/whisper/1/

# Installer les dépendances Python nécessaires
RUN pip3 install --upgrade pip && \
    pip3 install transformers==4.34.0 \
                 torch==2.0.1 \
                 torchaudio==2.0.2 \
                 numpy==1.24.3 \
                 ffmpeg-python==0.2.0 \
                 accelerate==0.23.0 \
                 sentencepiece==0.1.99

# Précharger et mettre en cache le modèle Whisper pour accélérer le premier chargement
RUN python3 -c "from transformers import pipeline; import torch; \
    pipeline('automatic-speech-recognition', 'openai/whisper-medium.en', torch_dtype=torch.float16)"

# Exposer les ports Triton
EXPOSE 8000 8001 8002

# CMD pour démarrer Triton avec le répertoire modèle approprié
CMD ["tritonserver", "--model-repository=/models"]