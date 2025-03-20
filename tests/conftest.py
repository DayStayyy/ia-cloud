import os
import json
import shutil
import pytest
import tempfile
from fastapi.testclient import TestClient
from fastapi import UploadFile, File
from unittest.mock import patch, MagicMock

# Assurez-vous que ces imports pointent vers vos modules réels
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from main import app, load_statistics, save_statistics


@pytest.fixture
def client():
    """
    Crée un client de test pour l'API FastAPI
    """
    with TestClient(app) as client:
        yield client


@pytest.fixture
def test_audio_file():
    """
    Crée un fichier audio temporaire pour les tests
    """
    # Crée un fichier temporaire
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_file.close()

    # Écrit des données factices
    with open(temp_file.name, "wb") as f:
        f.write(b"\x00" * 1000)  # Fichier WAV factice

    yield temp_file.name

    # Nettoie le fichier après le test
    if os.path.exists(temp_file.name):
        os.unlink(temp_file.name)


@pytest.fixture
def test_video_file():
    """
    Crée un fichier vidéo temporaire pour les tests
    """
    # Crée un fichier temporaire
    temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    temp_file.close()

    # Écrit des données factices
    with open(temp_file.name, "wb") as f:
        f.write(b"\x00" * 1000)  # Fichier MP4 factice

    yield temp_file.name

    # Nettoie le fichier après le test
    if os.path.exists(temp_file.name):
        os.unlink(temp_file.name)


@pytest.fixture
def mock_processing_result():
    """
    Retourne un résultat de traitement fictif
    """
    return {
        "result_url": "/results/test-id/video_with_subtitles.mp4",
        "download_url": "/download/test-id/video_with_subtitles.mp4",
        "model_used": "whisper-small",
        "speaker_identification": False
    }


@pytest.fixture
def temp_stats_dir():
    """
    Crée un répertoire temporaire pour les statistiques
    """
    temp_dir = tempfile.mkdtemp()
    original_stats_file = app.state.STATS_FILE if hasattr(app.state, "STATS_FILE") else "stats/statistics.json"

    # Sauvegarder le chemin original
    app.state.ORIGINAL_STATS_FILE = original_stats_file

    # Définir le nouveau chemin pour les tests
    stats_file = os.path.join(temp_dir, "statistics.json")
    app.state.STATS_FILE = stats_file

    # Initialiser les statistiques test
    initial_stats = {
        "avg_processing_time": 0,
        "total_inferences": 0,
        "inferences_by_type": {"1": 0, "2": 0, "3": 0, "4": 0},
        "inferences_by_model": {"whisper-small": 0},
        "pending_processes": 0,
        "pending_by_type": {"1": 0, "2": 0, "3": 0, "4": 0},
        "failed_processes": 0,
        "failed_by_type": {"1": 0, "2": 0, "3": 0, "4": 0},
        "processing_times": []
    }

    os.makedirs(os.path.dirname(stats_file), exist_ok=True)
    with open(stats_file, "w") as f:
        json.dump(initial_stats, f)

    yield stats_file

    # Restaurer le chemin original
    if hasattr(app.state, "ORIGINAL_STATS_FILE"):
        app.state.STATS_FILE = app.state.ORIGINAL_STATS_FILE
        delattr(app.state, "ORIGINAL_STATS_FILE")

    # Nettoyer le répertoire temporaire
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_process_video():
    """
    Mock pour la fonction process_video
    """
    with patch('main.process_video') as mock:
        mock.return_value = (
            {
                "result_url": "/results/test-id/video_with_subtitles.mp4",
                "download_url": "/download/test-id/video_with_subtitles.mp4",
                "model_used": "whisper-small",
                "speaker_identification": False
            },
            1.5,  # Temps de traitement fictif
            "whisper-small"  # Modèle utilisé
        )
        yield mock