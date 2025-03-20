import os
import json
import pytest
from unittest.mock import patch, mock_open
from main import load_statistics, save_statistics, update_statistics


def test_load_statistics_success(temp_stats_dir):
    """
    Teste que load_statistics charge correctement les statistiques existantes
    """
    # Préparer des statistiques test
    test_stats = {
        "avg_processing_time": 10.5,
        "total_inferences": 42,
        "inferences_by_type": {"1": 20, "2": 15, "3": 5, "4": 2},
        "inferences_by_model": {"whisper-small": 30, "whisper-medium": 12},
        "pending_processes": 3,
        "pending_by_type": {"1": 1, "2": 1, "3": 1, "4": 0},
        "failed_processes": 2,
        "failed_by_type": {"1": 1, "2": 0, "3": 0, "4": 1},
        "processing_times": [10.2, 9.8, 11.5]
    }

    with open(temp_stats_dir, "w") as f:
        json.dump(test_stats, f)

    # Charger les statistiques
    with patch("main.STATS_FILE", temp_stats_dir):
        loaded_stats = load_statistics()

    # Vérifier qu'elles sont chargées correctement
    assert loaded_stats == test_stats


def test_load_statistics_file_not_found():
    """
    Teste que load_statistics retourne des statistiques par défaut si le fichier n'existe pas
    """
    # Utiliser un chemin qui n'existe pas
    non_existent_file = "/path/that/does/not/exist/stats.json"

    # Charger les statistiques
    with patch("main.STATS_FILE", non_existent_file):
        loaded_stats = load_statistics()

    # Vérifier que des statistiques par défaut sont retournées
    assert loaded_stats["avg_processing_time"] == 0
    assert loaded_stats["total_inferences"] == 0
    assert "inferences_by_type" in loaded_stats
    assert "pending_processes" in loaded_stats
    assert "failed_processes" in loaded_stats
    assert "processing_times" in loaded_stats


def test_load_statistics_json_decode_error():
    """
    Teste que load_statistics gère correctement les erreurs de décodage JSON
    """
    # Simuler un fichier de statistiques mal formé
    invalid_json = "{ this is not valid json }"

    # Mock open pour retourner un contenu invalide
    with patch("builtins.open", mock_open(read_data=invalid_json)):
        with patch("os.path.exists", return_value=True):
            with patch("main.STATS_FILE", "stats.json"):
                loaded_stats = load_statistics()

    # Vérifier que des statistiques par défaut sont retournées
    assert loaded_stats["avg_processing_time"] == 0
    assert loaded_stats["total_inferences"] == 0


def test_save_statistics(temp_stats_dir):
    """
    Teste que save_statistics sauvegarde correctement les statistiques
    """
    # Statistiques à sauvegarder
    test_stats = {
        "avg_processing_time": 15.7,
        "total_inferences": 100,
        "inferences_by_type": {"1": 50, "2": 30, "3": 15, "4": 5},
        "inferences_by_model": {"whisper-small": 70, "whisper-medium": 30},
        "pending_processes": 5,
        "pending_by_type": {"1": 2, "2": 2, "3": 1, "4": 0},
        "failed_processes": 3,
        "failed_by_type": {"1": 2, "2": 1, "3": 0, "4": 0},
        "processing_times": [15.2, 16.3, 15.6]
    }

    # Sauvegarder les statistiques
    with patch("main.STATS_FILE", temp_stats_dir):
        save_statistics(test_stats)

    # Vérifier qu'elles sont correctement sauvegardées
    with open(temp_stats_dir, "r") as f:
        saved_stats = json.load(f)

    assert saved_stats == test_stats


def test_update_statistics_success(temp_stats_dir):
    """
    Teste que update_statistics met correctement à jour les statistiques après un traitement réussi
    """
    # Préparer les statistiques initiales
    initial_stats = {
        "avg_processing_time": 10.0,
        "total_inferences": 10,
        "inferences_by_type": {"1": 5, "2": 3, "3": 2, "4": 0},
        "inferences_by_model": {"whisper-small": 8, "whisper-medium": 2},
        "pending_processes": 2,
        "pending_by_type": {"1": 1, "2": 1, "3": 0, "4": 0},
        "failed_processes": 1,
        "failed_by_type": {"1": 0, "2": 0, "3": 1, "4": 0},
        "processing_times": [9.5, 10.5]
    }

    with open(temp_stats_dir, "w") as f:
        json.dump(initial_stats, f)

    # Mettre à jour les statistiques
    with patch("main.STATS_FILE", temp_stats_dir):
        update_statistics(1, 12.0, "whisper-small", success=True)

    # Charger les statistiques mises à jour
    with open(temp_stats_dir, "r") as f:
        updated_stats = json.load(f)

    # Vérifier les mises à jour
    assert updated_stats["total_inferences"] == 11  # +1
    assert updated_stats["inferences_by_type"]["1"] == 6  # +1
    assert updated_stats["inferences_by_model"]["whisper-small"] == 9  # +1
    assert updated_stats["pending_processes"] == 1  # -1
    assert updated_stats["pending_by_type"]["1"] == 0  # -1
    assert 12.0 in updated_stats["processing_times"]  # Ajout du nouveau temps


def test_update_statistics_failure(temp_stats_dir):
    """
    Teste que update_statistics met correctement à jour les statistiques après un traitement échoué
    """
    # Préparer les statistiques initiales
    initial_stats = {
        "avg_processing_time": 10.0,
        "total_inferences": 10,
        "inferences_by_type": {"1": 5, "2": 3, "3": 2, "4": 0},
        "inferences_by_model": {"whisper-small": 8, "whisper-medium": 2},
        "pending_processes": 2,
        "pending_by_type": {"1": 1, "2": 1, "3": 0, "4": 0},
        "failed_processes": 1,
        "failed_by_type": {"1": 0, "2": 0, "3": 1, "4": 0},
        "processing_times": [9.5, 10.5]
    }

    with open(temp_stats_dir, "w") as f:
        json.dump(initial_stats, f)

    # Mettre à jour les statistiques pour un échec
    with patch("main.STATS_FILE", temp_stats_dir):
        update_statistics(1, 0, "whisper-small", success=False)

    # Charger les statistiques mises à jour
    with open(temp_stats_dir, "r") as f:
        updated_stats = json.load(f)

    # Vérifier les mises à jour
    assert updated_stats["total_inferences"] == 10  # Inchangé
    assert updated_stats["inferences_by_type"]["1"] == 5  # Inchangé
    assert updated_stats["inferences_by_model"]["whisper-small"] == 8  # Inchangé
    assert updated_stats["pending_processes"] == 1  # -1
    assert updated_stats["pending_by_type"]["1"] == 0  # -1
    assert updated_stats["failed_processes"] == 2  # +1
    assert updated_stats["failed_by_type"]["1"] == 1  # +1


def test_update_statistics_large_processing_time(temp_stats_dir):
    """
    Teste que update_statistics ignore les temps de traitement aberrants
    """
    # Préparer les statistiques initiales
    initial_stats = {
        "avg_processing_time": 10.0,