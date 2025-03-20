# Application de Sous-titrage Automatique

Ce projet est une application d'industrialisation d'un pipeline d'IA dans le cloud qui permet de générer automatiquement des sous-titres pour des fichiers vidéo et audio.

## Fonctionnalités

- Traitement de fichiers vidéo (.mp4) et audio (.wav)
- Plusieurs options de sortie:
  - Vidéo avec sous-titres intégrés
  - Vidéo + fichier de sous-titres séparé
  - Sous-titres uniquement (format SRT)
  - Texte brut (transcription sans horodatage)
- Interface utilisateur intuitive avec prévisualisation des fichiers
- Tableau de bord des statistiques pour le suivi des performances
- Robustesse avec gestion des erreurs avancée
- Architecture scalable pouvant supporter jusqu'à 10 utilisateurs en parallèle

## Architecture

Le projet est composé de deux services principaux:

1. **Service API** - Application FastAPI qui gère l'interface utilisateur, le traitement des fichiers et la génération des résultats
2. **Service Triton** - Serveur d'inférence Triton pour l'exécution optimisée du modèle Whisper

```
┌─────────────┐     ┌───────────────┐     ┌────────────────┐
│ Utilisateur │────▶│ Service API   │────▶│ Service Triton │
│             │◀────│ (FastAPI)     │◀────│ (Inference)    │
└─────────────┘     └───────────────┘     └────────────────┘
```

## Prérequis

- Docker et Docker Compose
- FFmpeg (pour le développement local)
- Au moins 8 Go de RAM disponible
- Espace disque de 80 Go minimum

## Installation

### Avec Docker (recommandé)

1. Clonez le dépôt:
   ```bash
    git clone git@github.com:DayStayyy/ia-cloud.git   
    cd ia-cloud
   ```

2. Lancez les conteneurs avec Docker Compose:
   ```bash
   docker-compose up --build
   ```

3. Accédez à l'application via votre navigateur:
   ```
   http://localhost:8003
   ```

### Installation manuelle (développement - optionnel)

Cette méthode est utile uniquement pour le développement sans Docker.

1. Clonez le dépôt:
   ```bash
   git clone https://github.com/votre-repo/application-sous-titrage.git
   cd application-sous-titrage
   ```

2. Créez un environnement virtuel et activez-le:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Sur Windows: venv\Scripts\activate
   ```

3. Installez les dépendances:
   ```bash
   pip install -r requirements.txt
   ```

4. Lancez l'application:
   ```bash
   cd api
   python -m uvicorn main_without_triton:app --host 0.0.0.0 --port 8003
   ```

> **Note:** L'installation manuelle n'est généralement pas nécessaire car Docker installe automatiquement toutes les dépendances requises.

## Utilisation

1. Accédez à l'interface web via http://localhost:8003
2. Téléchargez un fichier vidéo (.mp4) ou audio (.wav)
3. Sélectionnez le type de sortie souhaité:
   - Vidéo avec sous-titres intégrés
   - Vidéo + sous-titres séparés
   - Sous-titres uniquement
   - Texte uniquement
4. Cliquez sur "Lancer le traitement"
5. Visualisez et téléchargez le résultat

## Administration
### Stopper le service 

```bash
docker-compose stop
```

### Redémarrer le service

```bash
docker-compose restart
```

### Vérifier les logs

```bash
docker-compose logs -f
```

### Accéder aux statistiques

Les statistiques sont disponibles dans l'interface utilisateur ou via l'API:
```
http://localhost:8003/stats
```
Ou
```
GET http://localhost:8003/api/statistics
GET http://localhost:8003/api/triton-statistics
```

## Structure du projet

```
application-sous-titrage/
├── api/
│   ├── main.py                 # Version principale avec Triton
│   ├── main_without_triton.py  # Version alternative sans Triton
│   ├── static/
│   │   └── js/
│   │       └── main.js         # JavaScript frontend
│   └── templates/
│       └── index.html          # Interface utilisateur
├── model_repository/
│   └── whisper/
│       ├── config.pbtxt        # Configuration Triton
│       └── 1/
│           └── model.py        # Modèle Python pour Triton
├── Dockerfile                  # Dockerfile pour Triton
├── Dockerfile.api              # Dockerfile pour l'API
├── docker-compose.yml          # Configuration Docker Compose
└── requirements.txt            # Dépendances Python
```

## Performance

- Temps de traitement: inférieur à la durée de l'échantillon fourni
- Capacité: jusqu'à 10 utilisateurs simultanés
- Utilisation des ressources: optimisée pour rester sous 10 000 euros/an d'infrastructure

## Dépannage

### Problème: Erreur de connexion à Triton

Vérifiez que le service Triton est bien en cours d'exécution:
```bash
docker-compose ps
```

Si nécessaire, redémarrez uniquement le service Triton:
```bash
docker-compose restart triton
```

### Problème: Échec du traitement des fichiers

Vérifiez les formats de fichiers supportés (.mp4 et .wav). Les autres formats ne sont pas pris en charge.

### Problème: Erreur de mémoire

Augmentez la mémoire allouée à Docker dans les paramètres Docker Desktop.

## Extension du projet

Pour les data scientists souhaitant ajouter de nouveaux modèles:

1. Ajoutez votre modèle dans le dossier `model_repository/`
2. Modifiez le fichier `config.pbtxt` correspondant
3. Mettez à jour l'API pour utiliser votre nouveau modèle

# Tests Unitaires pour l'Application de Sous-titrage

Ce document explique comment installer les dépendances nécessaires et exécuter les tests unitaires pour l'application de sous-titrage automatique.

## Prérequis

Pour exécuter les tests unitaires, vous devez avoir les éléments suivants installés :

- Python 3.8 ou supérieur
- pip (gestionnaire de paquets Python)

## Installation des dépendances

1. Installez les dépendances nécessaires pour les tests :

```bash
pip install pytest pytest-asyncio httpx pytest-cov
```

Ces paquets sont nécessaires pour :
- `pytest` : Framework de test principal
- `pytest-asyncio` : Support pour les tests asynchrones
- `httpx` : Client HTTP utilisé par le testclient de FastAPI
- `pytest-cov` : Génération de rapports de couverture de code

## Structure des tests

Les tests sont organisés dans le dossier `tests/` avec la structure suivante :
```
tests/
├── __init__.py
├── conftest.py                  # Configuration pytest et fixtures partagées
├── test_api.py                  # Tests des routes API
├── test_srt_generation.py       # Tests de génération des sous-titres
├── test_statistics.py           # Tests des fonctions de statistiques
└── test_utils.py                # Tests des fonctions utilitaires
```

## Exécution des tests

### Pour exécuter tous les tests :

```bash
pytest
```

### Pour exécuter un fichier de test spécifique :

```bash
pytest tests/test_api.py
```

### Pour exécuter un test spécifique :

```bash
pytest tests/test_api.py::test_home_route
```

### Pour exécuter les tests avec le rapport de couverture :

```bash
pytest --cov=app tests/
```

Cela générera un rapport indiquant le pourcentage de code couvert par les tests.

### Pour générer un rapport de couverture HTML détaillé :

```bash
pytest --cov=app --cov-report=html tests/
```

Le rapport sera généré dans le dossier `htmlcov/`. Ouvrez `htmlcov/index.html` dans votre navigateur pour voir les détails.

## Dépannage des tests

### Erreur "ModuleNotFoundError"

Si vous rencontrez des erreurs d'importation comme `ModuleNotFoundError`, assurez-vous que le répertoire racine du projet est dans votre `PYTHONPATH` :

```bash
# Sur Linux/Mac
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Sur Windows (PowerShell)
$env:PYTHONPATH += ";$pwd"
```

### Erreur avec Docker

Pour exécuter les tests dans un environnement Docker, utilisez la commande suivante :

```bash
docker-compose run --rm api pytest
```

Cela lance les tests dans le conteneur API, assurant que l'environnement est identique à celui de production.

## Intégration Continue

Si vous utilisez un système d'intégration continue (CI), vous pouvez ajouter cette commande à votre workflow pour exécuter automatiquement les tests à chaque commit :

```bash
python -m pytest tests/ --cov=app --cov-report=xml
```

Le fichier `coverage.xml` généré peut être utilisé par des outils comme SonarQube ou Codecov pour suivre la couverture du code.