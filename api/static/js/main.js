document.addEventListener('DOMContentLoaded', function() {
    // Gestion de l'aperçu du fichier téléchargé
    const fileInput = document.getElementById('fileInput');
    const videoPreview = document.getElementById('videoPreview');
    const audioPreview = document.getElementById('audioPreview');

    fileInput.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (!file) return;

        // Cacher les deux previews d'abord
        videoPreview.style.display = 'none';
        audioPreview.style.display = 'none';

        // Vérifier le type de fichier et af
        // ficher le preview approprié
        if (file.type.startsWith('video/')) {
            const videoURL = URL.createObjectURL(file);
            videoPreview.src = videoURL;
            videoPreview.style.display = 'block';
            
            // Activer/désactiver les options de sortie en fonction du type de fichier
            document.getElementById('videoEmbedded').disabled = false;
            document.getElementById('videoMetadata').disabled = false;
        } else if (file.type === 'audio/wav' || file.type === 'audio/x-wav') {
            const audioURL = URL.createObjectURL(file);
            audioPreview.src = audioURL;
            audioPreview.style.display = 'block';
            
            // Désactiver les options qui ne sont pas disponibles pour l'audio
            document.getElementById('videoEmbedded').disabled = true;
            document.getElementById('videoMetadata').disabled = true;
            
            // Sélectionner une option disponible
            document.getElementById('subtitlesOnly').checked = true;
        }
    });

    // Gestion du formulaire d'upload
    const uploadForm = document.getElementById('uploadForm');
    const processingLoader = document.getElementById('processingLoader');
    const resultSection = document.getElementById('resultSection');
    const resultContent = document.getElementById('resultContent');
    const downloadResult = document.getElementById('downloadResult');

    uploadForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        // Vérification du fichier
        const file = fileInput.files[0];
        if (!file) {
            alert('Veuillez sélectionner un fichier');
            return;
        }
        
        // Récupération du type de résultat souhaité
        const outputTypeRadios = document.getElementsByName('outputType');
        let selectedOutputType;
        
        for (const radio of outputTypeRadios) {
            if (radio.checked) {
                selectedOutputType = radio.value;
                break;
            }
        }
        
        // Préparation des données à envoyer
        const formData = new FormData();
        formData.append('video', file); // Le nom du paramètre doit correspondre à celui dans FastAPI
        formData.append('type_of_request', selectedOutputType);
        
        // Afficher le loader
        processingLoader.style.display = 'block';
        
        try {
            // Envoi de la requête au serveur
            const response = await fetch('/upload-video', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error(`Erreur HTTP: ${response.status}`);
            }
            
            const data = await response.json();
            
            // Masquer le loader
            processingLoader.style.display = 'none';
            
            // Afficher la section de résultat
            resultSection.style.display = 'block';
            
            // Traitement de la réponse en fonction du type de sortie demandé
            displayResult(data, selectedOutputType);
            
            // Mettre à jour le lien de téléchargement
            if (data.download_url) {
                downloadResult.href = data.download_url;
                downloadResult.style.display = 'inline-block';
            } else {
                downloadResult.style.display = 'none';
            }
            
            // Rafraîchir les statistiques
            fetchStatistics();
            
        } catch (error) {
            console.error('Erreur lors du traitement:', error);
            alert('Une erreur est survenue lors du traitement du fichier: ' + error.message);
            processingLoader.style.display = 'none';
        }
    });

    // Fonction pour afficher le résultat selon le type demandé
    function displayResult(data, outputType) {
        resultContent.innerHTML = '';
        
        switch (parseInt(outputType)) {
            case 1: // Vidéo avec sous-titres intégrés
                resultContent.innerHTML = `
                    <div class="alert alert-success">Traitement terminé avec succès!</div>
                    <video controls width="100%" src="${data.result_url}"></video>
                `;
                break;
            
            case 2: // Vidéo + sous-titres séparés
                resultContent.innerHTML = `
                    <div class="alert alert-success">Traitement terminé avec succès!</div>
                    <video controls width="100%" src="${data.video_url}"></video>
                    <div class="mt-3">
                        <h5>Sous-titres:</h5>
                        <pre class="bg-light p-3 rounded">${data.subtitles}</pre>
                    </div>
                `;
                break;
            
            case 3: // Sous-titres uniquement
                resultContent.innerHTML = `
                    <div class="alert alert-success">Traitement terminé avec succès!</div>
                    <h5>Sous-titres:</h5>
                    <pre class="bg-light p-3 rounded">${data.subtitles}</pre>
                `;
                break;
            
            case 4: // Texte uniquement
                resultContent.innerHTML = `
                    <div class="alert alert-success">Traitement terminé avec succès!</div>
                    <h5>Transcription complète:</h5>
                    <div class="bg-light p-3 rounded">
                        <p>${data.text}</p>
                    </div>
                `;
                break;
            
            default:
                resultContent.innerHTML = `
                    <div class="alert alert-warning">Type de résultat non reconnu.</div>
                    <pre>${JSON.stringify(data, null, 2)}</pre>
                `;
        }
    }

    // Récupération et affichage des statistiques
    const refreshStatsButton = document.getElementById('refreshStats');
    
    refreshStatsButton.addEventListener('click', fetchStatistics);
    
    // Fonction pour récupérer les statistiques
    async function fetchStatistics() {
        try {
            const response = await fetch('/api/statistics');
            if (!response.ok) {
                throw new Error(`Erreur HTTP: ${response.status}`);
            }
            
            const stats = await response.json();
            
            // Mise à jour des informations dans l'UI
            document.getElementById('avgProcessingTime').textContent = `${stats.avg_processing_time.toFixed(2)} secondes`;
            document.getElementById('totalInferences').textContent = stats.total_inferences;
            document.getElementById('pendingProcesses').textContent = stats.pending_processes;
            document.getElementById('failedProcesses').textContent = stats.failed_processes;
            
        } catch (error) {
            console.error('Erreur lors de la récupération des statistiques:', error);
        }
    }
    
    // Charger les statistiques au chargement de la page
    fetchStatistics();
});
