# Image NVIDIA Triton de base
FROM nvcr.io/nvidia/tritonserver:23.10-py3

# Désactiver l'utilisation des GPU
ENV CUDA_VISIBLE_DEVICES=""

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

# Copier seulement les requirements et les installer d'abord
# Cela permet de mettre en cache cette couche tant que les requirements ne changent pas
COPY requirements.txt /tmp/
RUN pip3 install --upgrade pip && \
    pip3 install -r /tmp/requirements.txt

RUN apt-get update && apt-get install -y ffmpeg


# Précharger et mettre en cache le modèle Whisper pour CPU
# Cette étape est longue et coûteuse, donc la mettre avant le code qui change souvent
RUN python3 -c "from transformers import pipeline; import torch; \
    pipeline('automatic-speech-recognition', 'openai/whisper-medium.en', device=-1)"

# Maintenant, copier les fichiers de configuration et modèle
# Ces fichiers changeront plus souvent, donc les mettre après les couches de cache
COPY model_repository/whisper/config.pbtxt /models/whisper/
COPY model_repository/whisper/1/model.py /models/whisper/1/

# Exposer les ports Triton
EXPOSE 8000 8001 8002

# CMD pour démarrer Triton avec le répertoire modèle approprié
CMD ["tritonserver", "--model-repository=/models"]