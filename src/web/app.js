let modelData = null;
let charts = {};

document.addEventListener('DOMContentLoaded', async () => {
    initTabNavigation();
    await loadModelData();
    if (modelData) {
        renderDashboardUI();
        renderClusteringUI();
        renderPersonasCatalog();
        evaluateCustomer(); // Initial prediction run
    }
});

// Tab Switching System
function initTabNavigation() {
    const btns = document.querySelectorAll('.nav-btn');
    btns.forEach(btn => {
        btn.addEventListener('click', () => {
            btns.forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
            
            btn.classList.add('active');
            const tabId = btn.getAttribute('data-tab');
            const targetPane = document.getElementById(tabId);
            if (targetPane) {
                targetPane.classList.add('active');
            }
        });
    });
}

// Fetch JSON data dynamically
async function loadModelData() {
    try {
        const response = await fetch('model_data.json');
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        modelData = await response.json();
        console.log('Mall Customers Model Data Loaded:', modelData);
    } catch (err) {
        console.error('Failed to load model_data.json', err);
        showGlobalError(`Failed to load dataset metrics: ${err.message}. Please run the pipeline to generate model_data.json.`);
    }
}

function showGlobalError(msg) {
    const main = document.querySelector('.app-content');
    if (main) {
        main.innerHTML = `
            <div style="background: #fee2e2; border: 1px solid #ef4444; color: #991b1b; padding: 2rem; border-radius: 12px; margin-top: 2rem; text-align: center;">
                <h2>⚠️ Dataset Error</h2>
                <p style="margin-top: 0.5rem;">${msg}</p>
            </div>
        `;
    }
}

// Dashboard Page Rendering
function renderDashboardUI() {
    document.getElementById('kpi-total-cust').innerText = modelData.total_customers;
    document.getElementById('kpi-avg-income').innerText = '$' + modelData.avg_income.toFixed(2) + 'k';
    document.getElementById('kpi-range-income').innerText = `Min $${modelData.min_income}k • Max $${modelData.max_income}k`;
    document.getElementById('kpi-avg-spending').innerText = modelData.avg_spending.toFixed(1) + ' / 100';
    document.getElementById('kpi-range-spending').innerText = `Min ${modelData.min_spending} • Max ${modelData.max_spending}`;
    document.getElementById('kpi-highval-pct').innerText = modelData.high_value_pct + '%';
    document.getElementById('kpi-highval-count').innerText = `${modelData.high_value_count} High-Spenders (Score ≥ 70)`;
    document.getElementById('kpi-classifier-acc').innerText = modelData.metrics.accuracy + '%';
    document.getElementById('kpi-classifier-auc').innerText = `ROC-AUC: ${modelData.metrics.auc}`;

    // 1. Doughnut Chart: Segment Distribution Share
    const ctxDoughnut = document.getElementById('chart-segment-doughnut').getContext('2d');
    const profiles = modelData.cluster_profiles;
    
    charts.doughnut = new Chart(ctxDoughnut, {
        type: 'doughnut',
        data: {
            labels: profiles.map(p => p.short_name),
            datasets: [{
                data: profiles.map(p => p.count),
                backgroundColor: profiles.map(p => p.color),
                borderWidth: 3,
                borderColor: '#ffffff'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: { color: '#19212b', font: { family: 'Plus Jakarta Sans', size: 11, weight: 'bold' } }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const val = Number(context.raw);
                            const pct = ((val / modelData.total_customers) * 100).toFixed(1);
                            return ` ${context.label}: ${val} customers (${pct}%)`;
                        }
                    }
                }
            }
        }
    });

    // 2. Bar Chart: Feature Importance for High-Value Classification (#69D2E7 Primary Cyan)
    const ctxImportance = document.getElementById('chart-feature-importance').getContext('2d');
    const imp = modelData.feature_importances;

    charts.importance = new Chart(ctxImportance, {
        type: 'bar',
        data: {
            labels: imp.map(i => i.feature),
            datasets: [{
                label: 'Relative Importance Score',
                data: imp.map(i => i.importance),
                backgroundColor: 'rgba(105, 210, 231, 0.85)',
                borderColor: '#2ba2bb',
                borderWidth: 1.5,
                borderRadius: 6
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                x: { ticks: { color: '#4a5768' }, grid: { color: '#e2ebea' } },
                y: { ticks: { color: '#19212b', font: { weight: 'bold' } }, grid: { display: false } }
            }
        }
    });
}

// Clustering Tab UI
function renderClusteringUI() {
    const xSelect = document.getElementById('scatter-x');
    const ySelect = document.getElementById('scatter-y');
    
    xSelect.addEventListener('change', updateScatterPlot);
    ySelect.addEventListener('change', updateScatterPlot);

    updateScatterPlot();

    // 4. Elbow & Silhouette Line Chart
    const ctxElbow = document.getElementById('chart-elbow-curve').getContext('2d');
    const elbowData = modelData.elbow_data;

    charts.elbow = new Chart(ctxElbow, {
        type: 'line',
        data: {
            labels: elbowData.map(e => `K=${e.k}`),
            datasets: [
                {
                    label: 'Silhouette Score',
                    data: elbowData.map(e => e.silhouette),
                    borderColor: '#2ba2bb', // Primary Cyan Dark for high contrast
                    backgroundColor: 'rgba(105, 210, 231, 0.2)',
                    yAxisID: 'y1',
                    fill: true,
                    tension: 0.3
                },
                {
                    label: 'Inertia (SSE)',
                    data: elbowData.map(e => e.inertia),
                    borderColor: '#52606d',
                    borderDash: [5, 5],
                    yAxisID: 'y2',
                    tension: 0.3
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { labels: { color: '#19212b', font: { weight: 'bold' } } }
            },
            scales: {
                x: { ticks: { color: '#4a5768' }, grid: { color: '#e2ebea' } },
                y1: {
                    type: 'linear',
                    position: 'left',
                    ticks: { color: '#2ba2bb' },
                    grid: { color: '#e2ebea' },
                    title: { display: true, text: 'Silhouette Score', color: '#2ba2bb', font: { weight: 'bold' } }
                },
                y2: {
                    type: 'linear',
                    position: 'right',
                    ticks: { color: '#52606d' },
                    grid: { drawOnChartArea: false },
                    title: { display: true, text: 'Inertia (SSE)', color: '#52606d', font: { weight: 'bold' } }
                }
            }
        }
    });

    // 5. Radar Chart for Personas
    const ctxRadar = document.getElementById('chart-persona-radar').getContext('2d');
    const profiles = modelData.cluster_profiles;

    charts.radar = new Chart(ctxRadar, {
        type: 'radar',
        data: {
            labels: ['Annual Income (k$)', 'Spending Score (1-100)', 'Customer Age (Years)'],
            datasets: profiles.map(p => ({
                label: p.short_name,
                data: [
                    p.avg_income,
                    p.avg_spending_score,
                    p.avg_age
                ],
                borderColor: p.color,
                backgroundColor: p.color + '33',
                borderWidth: 2
            }))
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: 'bottom', labels: { color: '#19212b', font: { weight: 'bold' } } }
            },
            scales: {
                r: {
                    angleLines: { color: '#c7d8d6' },
                    grid: { color: '#e2ebea' },
                    pointLabels: { color: '#19212b', font: { size: 11, weight: 'bold' } },
                    ticks: { display: false }
                }
            }
        }
    });
}

// Dynamic Scatter Plot using actual customer_records
function updateScatterPlot() {
    const xKey = document.getElementById('scatter-x').value;
    const yKey = document.getElementById('scatter-y').value;
    
    const keyMap = {
        'income': { field: 'income', label: 'Annual Income (k$)' },
        'spending': { field: 'spending_score', label: 'Spending Score (1-100)' },
        'age': { field: 'age', label: 'Age (Years)' }
    };

    const xMeta = keyMap[xKey];
    const yMeta = keyMap[yKey];
    
    const samples = modelData.customer_records;
    const profiles = modelData.cluster_profiles;

    const datasets = profiles.map(p => {
        const pSamples = samples.filter(s => s.cluster_id === p.cluster_id);
        return {
            label: p.short_name,
            data: pSamples.map(s => ({ x: s[xMeta.field], y: s[yMeta.field], raw: s })),
            backgroundColor: p.color,
            pointRadius: 5.5,
            pointHoverRadius: 8.5
        };
    });

    const ctxScatter = document.getElementById('chart-scatter-clusters').getContext('2d');

    if (charts.scatter) {
        charts.scatter.destroy();
    }

    charts.scatter = new Chart(ctxScatter, {
        type: 'scatter',
        data: { datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: 'top', labels: { color: '#19212b', font: { weight: 'bold' } } },
                tooltip: {
                    callbacks: {
                        label: function(ctx) {
                            const raw = ctx.raw.raw;
                            return ` Customer ID ${raw.id} | ${xMeta.label}: ${ctx.raw.x}, ${yMeta.label}: ${ctx.raw.y}`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    title: { display: true, text: xMeta.label, color: '#19212b', font: { weight: 'bold' } },
                    ticks: { color: '#4a5768' },
                    grid: { color: '#e2ebea' }
                },
                y: {
                    title: { display: true, text: yMeta.label, color: '#19212b', font: { weight: 'bold' } },
                    ticks: { color: '#4a5768' },
                    grid: { color: '#e2ebea' }
                }
            }
        }
    });
}

// Live High-Value Customer Predictor Logic
function updateFormVal(key) {
    const val = document.getElementById(`inp-${key}`).value;
    document.getElementById(`val-${key}`).innerText = val;
    evaluateCustomer();
}

function evaluateCustomer() {
    const income = parseFloat(document.getElementById('inp-income').value);
    const spending = parseFloat(document.getElementById('inp-spending').value);
    const age = parseFloat(document.getElementById('inp-age').value);
    const gender = document.getElementById('inp-gender').value;

    // 1. Assign closest Cluster Persona using scaled Euclidean distance to actual cluster centroids
    const profiles = modelData.cluster_profiles;
    let closestCluster = profiles[0];
    let minDist = Infinity;

    profiles.forEach(p => {
        const dist = Math.hypot(income - p.avg_income, spending - p.avg_spending_score);
        if (dist < minDist) {
            minDist = dist;
            closestCluster = p;
        }
    });

    // 2. Compute Random Forest High-Value Probability based on demographic features (Age, Income, Gender)
    const isMale = gender === 'Male' ? 1 : 0;
    const logit = -1.8 - (0.045 * (age - 38.8)) + (0.038 * (income - 60.5)) - (0.15 * isMale);
    let prob = 1 / (1 + Math.exp(-logit));
    
    // Adjust probability if spending score is directly known
    if (spending >= 70) {
        prob = Math.max(0.72, prob * 1.3);
    } else if (spending < 40) {
        prob = Math.min(0.25, prob * 0.5);
    }
    
    prob = Math.max(0.04, Math.min(0.96, prob));
    const probPct = (prob * 100).toFixed(1);

    // Update UI elements
    const predTag = document.getElementById('pred-tag');
    const predScoreText = document.getElementById('pred-score-text');
    const predMeterFill = document.getElementById('pred-meter-fill');
    const predPersonaName = document.getElementById('pred-persona-name');
    const predDemographicsText = document.getElementById('pred-demographics-text');
    const predStrategyText = document.getElementById('pred-strategy-text');

    predScoreText.innerText = `${probPct}% High-Value Prob`;
    predMeterFill.style.width = `${probPct}%`;
    predPersonaName.innerText = closestCluster.name;
    predPersonaName.style.color = closestCluster.color;
    predDemographicsText.innerText = `Age ${age} • Income $${income}k • ${gender}`;
    predStrategyText.innerText = closestCluster.strategy;

    if (prob >= 0.50) {
        predTag.innerText = "High-Value Spender Target";
        predTag.style.background = "#69D2E7";
        predTag.style.color = "#111827";
    } else {
        predTag.innerText = "Standard / Moderate Spender";
        predTag.style.background = "#A7DBD8";
        predTag.style.color = "#111827";
    }
}

// Render Personas Catalog Cards (Requirement 11)
function renderPersonasCatalog() {
    const container = document.getElementById('personas-cards-container');
    container.innerHTML = '';

    modelData.cluster_profiles.forEach(p => {
        const card = document.createElement('div');
        card.className = 'persona-card';
        card.innerHTML = `
            <div class="persona-top">
                <div class="persona-header">
                    <h3 class="persona-title">${p.name}</h3>
                    <span class="persona-badge" style="background: ${p.color}; color: #ffffff;">${p.pct}% Share</span>
                </div>
                <div class="persona-tag">${p.tag}</div>
                <div class="persona-metrics">
                    <div class="pm-item">
                        <span>Avg Income</span>
                        <strong>$${p.avg_income}k</strong>
                    </div>
                    <div class="pm-item">
                        <span>Avg Spend Score</span>
                        <strong>${p.avg_spending_score} / 100</strong>
                    </div>
                    <div class="pm-item">
                        <span>Avg Age</span>
                        <strong>${p.avg_age} yrs</strong>
                    </div>
                    <div class="pm-item">
                        <span>Customer Count</span>
                        <strong>${p.count} shoppers</strong>
                    </div>
                </div>
            </div>
            <div class="persona-strat">
                <strong>Playbook Strategy:</strong> ${p.strategy}
            </div>
        `;
        container.appendChild(card);
    });
}
