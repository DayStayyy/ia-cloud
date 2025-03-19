document.addEventListener('DOMContentLoaded', function() {
    // Références aux éléments du DOM
    const avgProcessingTimeEl = document.getElementById('avgProcessingTime');
    const totalInferencesEl = document.getElementById('totalInferences');
    const pendingProcessesEl = document.getElementById('pendingProcesses');
    const failedProcessesEl = document.getElementById('failedProcesses');
    const lastUpdateEl = document.getElementById('lastUpdate');
    const refreshStatsButton = document.getElementById('refreshStats');
    const statsTableBody = document.getElementById('statsTableBody');

    // Variables pour les graphiques
    let inferencesByTypeChart = null;
    let statusChart = null;

    // Initialisation des graphiques
    function initCharts() {
        // Graphique des inférences par type
        const inferencesByTypeCtx = document.getElementById('inferencesByTypeChart').getContext('2d');
        inferencesByTypeChart = new Chart(inferencesByTypeCtx, {
            type: 'bar',
            data: {
                labels: [
                    'Vidéo avec sous-titres',
                    'Vidéo + sous-titres séparés',
                    'Sous-titres uniquement',
                    'Texte uniquement'
                ],
                datasets: [{
                    label: 'Nombre d\'inférences',
                    data: [0, 0, 0, 0],
                    backgroundColor: [
                        'rgba(54, 162, 235, 0.7)',
                        'rgba(75, 192, 192, 0.7)',
                        'rgba(153, 102, 255, 0.7)',
                        'rgba(255, 159, 64, 0.7)'
                    ],
                    borderColor: [
                        'rgba(54, 162, 235, 1)',
                        'rgba(75, 192, 192, 1)',
                        'rgba(153, 102, 255, 1)',
                        'rgba(255, 159, 64, 1)'
                    ],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            precision: 0
                        }
                    }
                }
            }
        });

        // Graphique de statut des traitements
        const statusCtx = document.getElementById('statusChart').getContext('2d');
        statusChart = new Chart(statusCtx, {
            type: 'doughnut',
            data: {
                labels: ['Réussis', 'En cours', 'Échoués'],
                datasets: [{
                    data: [0, 0, 0],
                    backgroundColor: [
                        'rgba(40, 167, 69, 0.7)',
                        'rgba(0, 123, 255, 0.7)',
                        'rgba(220, 53, 69, 0.7)'
                    ],
                    borderColor: [
                        'rgba(40, 167, 69, 1)',
                        'rgba(0, 123, 255, 1)',
                        'rgba(220, 53, 69, 1)'
                    ],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'right'
                    }
                }
            }
        });
    }

    // Fonction pour mettre à jour le tableau des statistiques
    function updateStatsTable(stats) {
        statsTableBody.innerHTML = '';

        const typeLabels = {
            '1': 'Vidéo avec sous-titres',
            '2': 'Vidéo + sous-titres séparés',
            '3': 'Sous-titres uniquement',
            '4': 'Texte uniquement'
        };

        for (const typeId in typeLabels) {
            const succeeded = stats.inferences_by_type[typeId];
            const pending = stats.pending_by_type[typeId];
            const failed = stats.failed_by_type[typeId];
            const total = succeeded + pending + failed;

            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${typeLabels[typeId]}</td>
                <td>${succeeded}</td>
                <td>${pending}</td>
                <td>${failed}</td>
                <td>${total}</td>
            `;

            statsTableBody.appendChild(row);
        }
    }

    // Fonction pour mettre à jour les graphiques
    function updateCharts(stats) {
        // Mise à jour du graphique des inférences par type
        inferencesByTypeChart.data.datasets[0].data = [
            stats.inferences_by_type['1'],
            stats.inferences_by_type['2'],
            stats.inferences_by_type['3'],
            stats.inferences_by_type['4']
        ];
        inferencesByTypeChart.update();

        // Mise à jour du graphique de statut
        statusChart.data.datasets[0].data = [
            stats.total_inferences,
            stats.pending_processes,
            stats.failed_processes
        ];
        statusChart.update();
    }

    // Fonction pour récupérer et afficher les statistiques
    async function fetchStatistics() {
        try {
            const response = await fetch('/api/statistics');
            if (!response.ok) {
                throw new Error(`Erreur HTTP: ${response.status}`);
            }

            const stats = await response.json();

            // Mise à jour des informations générales
            // Conversion du temps moyen de secondes en minutes
            const avgTimeInMinutes = stats.avg_processing_time / 60;
            avgProcessingTimeEl.textContent = avgTimeInMinutes.toFixed(2);
            totalInferencesEl.textContent = stats.total_inferences;
            pendingProcessesEl.textContent = stats.pending_processes;
            failedProcessesEl.textContent = stats.failed_processes;

            // Mise à jour de l'heure de dernière actualisation
            const now = new Date();
            lastUpdateEl.textContent = now.toLocaleString();

            // Mise à jour du tableau et des graphiques
            updateStatsTable(stats);
            updateCharts(stats);

        } catch (error) {
            console.error('Erreur lors de la récupération des statistiques:', error);
            alert('Impossible de récupérer les statistiques. Veuillez réessayer plus tard.');
        }
    }

    // Initialiser les graphiques au chargement de la page
    initCharts();

    // Charger les statistiques initiales
    fetchStatistics();

    // Configurer le rafraîchissement automatique toutes les 30 secondes
    setInterval(fetchStatistics, 30000);

    // Configurer le bouton de rafraîchissement
    refreshStatsButton.addEventListener('click', fetchStatistics);
});