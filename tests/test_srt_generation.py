import pytest
from main import format_time, generate_srt


def test_format_time():
    """
    Teste que la fonction format_time convertit correctement les secondes au format SRT
    """
    # Test des valeurs simples
    assert format_time(0) == "00:00:00,000"
    assert format_time(10) == "00:00:10,000"
    assert format_time(60) == "00:01:00,000"
    assert format_time(3600) == "01:00:00,000"

    # Test avec des millisecondes
    assert format_time(10.5) == "00:00:10,500"
    assert format_time(10.125) == "00:00:10,125"

    # Test avec des valeurs complexes
    assert format_time(3661.75) == "01:01:01,750"  # 1h 1min 1.75s
    assert format_time(7322.42) == "02:02:02,420"  # 2h 2min 2.42s


def test_generate_srt_with_chunks():
    """
    Teste que generate_srt génère correctement des sous-titres à partir de résultats avec des chunks
    """
    # Simuler un résultat de transcription avec chunks
    result = {
        "text": "Voici un exemple de transcription complète.",
        "chunks": [
            {"text": "Voici un exemple", "timestamp": [0.0, 2.0]},
            {"text": "de transcription", "timestamp": [2.2, 4.0]},
            {"text": "complète.", "timestamp": [4.2, 5.0]}
        ]
    }

    srt = generate_srt(result)

    # Vérifier le format général
    assert "1\n00:00:00,000 --> 00:00:02,000\nVoici un exemple\n\n" in srt
    assert "2\n00:00:02,200 --> 00:00:04,000\nde transcription\n\n" in srt
    assert "3\n00:00:04,200 --> 00:00:05,000\ncomplète.\n\n" in srt


def test_generate_srt_with_segments():
    """
    Teste que generate_srt génère correctement des sous-titres à partir de résultats avec des segments
    """
    # Simuler un résultat de transcription avec segments
    result = {
        "text": "Voici un exemple de transcription complète.",
        "segments": [
            {"text": "Voici un exemple", "start": 0.0, "end": 2.0},
            {"text": "de transcription", "start": 2.2, "end": 4.0},
            {"text": "complète.", "start": 4.2, "end": 5.0}
        ]
    }

    srt = generate_srt(result)

    # Vérifier le format général
    assert "1\n00:00:00,000 --> 00:00:02,000\nVoici un exemple\n\n" in srt
    assert "2\n00:00:02,200 --> 00:00:04,000\nde transcription\n\n" in srt
    assert "3\n00:00:04,200 --> 00:00:05,000\ncomplète.\n\n" in srt


def test_generate_srt_without_segments():
    """
    Teste que generate_srt génère un segment unique quand il n'y a pas de segments détaillés
    """
    # Simuler un résultat de transcription sans segments
    result = {
        "text": "Voici un exemple de transcription complète."
    }

    srt = generate_srt(result)

    # Vérifier le format général
    assert "1\n00:00:00,000 --> 00:01:00,000\nVoici un exemple de transcription complète.\n\n" in srt


def test_generate_srt_with_overlapping_segments():
    """
    Teste que generate_srt gère correctement les segments qui se chevauchent
    """
    # Simuler un résultat avec des chunks qui ont des timestamps incohérents
    result = {
        "text": "Exemple avec timestamp problématique.",
        "chunks": [
            {"text": "Exemple avec", "timestamp": [0.0, 2.0]},
            # Ce chunk a un timestamp de début > fin, simulant un nouveau segment
            {"text": "timestamp", "timestamp": [29.0, 28.0]},
            {"text": "problématique.", "timestamp": [2.0, 3.0]}
        ]
    }

    srt = generate_srt(result)

    # Vérifier que le second segment a été ignoré/traité correctement
    segments = srt.split("\n\n")
    # On devrait avoir deux segments valides
    assert len([s for s in segments if s.strip()]) == 2


def test_generate_srt_with_empty_text():
    """
    Teste que generate_srt gère correctement les segments avec un texte vide
    """
    # Simuler un résultat avec un segment sans texte
    result = {
        "text": "Exemple avec un segment vide.",
        "chunks": [
            {"text": "Exemple avec", "timestamp": [0.0, 2.0]},
            {"text": "", "timestamp": [2.2, 3.0]},  # Segment vide
            {"text": "un segment vide.", "timestamp": [3.2, 5.0]}
        ]
    }

    srt = generate_srt(result)

    # Vérifier que le second segment est inclus mais vide
    assert "2\n00:00:02,200 --> 00:00:03,000\n\n\n" in srt


def test_generate_srt_with_speaker_identification():
    """
    Teste que generate_srt préserve les identifications de locuteurs si présentes
    """
    # Simuler un résultat avec des identifications de locuteurs
    result = {
        "text": "Alice: Bonjour. Bob: Comment ça va?",
        "chunks": [
            {"text": "Alice: Bonjour.", "timestamp": [0.0, 2.0]},
            {"text": "Bob: Comment ça va?", "timestamp": [2.5, 4.0]}
        ]
    }

    srt = generate_srt(result)

    # Vérifier que les identifications sont préservées
    assert "Alice: Bonjour." in srt
    assert "Bob: Comment ça va?" in srt