import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import json

def render(df_clustered, result, k_value, features_cols):
    """Render Analysis tab - Hybrid Streamlit + HTML"""
    
    # ==================== PREPARE DATA ====================
    # Get cluster centers
    centers = result['scaler'].inverse_transform(result['kmeans'].cluster_centers_)
    
    # Convert to list of dicts for JSON
    centers_data = []
    for i in range(k_value):
        center_dict = {'cluster': i}
        for j, col in enumerate(features_cols):
            center_dict[col] = float(centers[i][j])
        centers_data.append(center_dict)
    
    # Get raw data for box plots (sample per cluster)
    raw_data = {}
    for cluster_num in range(k_value):
        cluster_df = df_clustered[df_clustered['Cluster'] == cluster_num]
        
        # Sample max 100 points per cluster untuk performa
        if len(cluster_df) > 100:
            cluster_df = cluster_df.sample(n=100, random_state=42)
        
        raw_data[f'cluster_{cluster_num}'] = {}
        for col in features_cols:
            raw_data[f'cluster_{cluster_num}'][col] = cluster_df[col].tolist()
    
    # Convert to JSON
    centers_json = json.dumps(centers_data)
    raw_data_json = json.dumps(raw_data)
    features_json = json.dumps(features_cols)
    
    # ==================== HTML COMPONENT ====================
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.27.0/plotly.min.js"></script>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}

            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
                background-color: transparent;
                padding: 0;
                color: #1E293B;
            }}

            .container {{
                max-width: 100%;
                margin: 0 auto;
            }}

            .card {{
                background-color: #FFFFFF;
                border-radius: 12px;
                padding: 1.5rem;
                margin-bottom: 1.5rem;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
            }}

            h3 {{
                color: #1E293B;
                font-size: 1.1rem;
                margin-bottom: 1rem;
                font-weight: 600;
            }}

            .two-column {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 1.5rem;
                margin-bottom: 1.5rem;
            }}

            .table-container {{
                overflow-x: auto;
                border-radius: 8px;
            }}

            table {{
                width: 100%;
                border-collapse: collapse;
                font-size: 0.85rem;
            }}

            th {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 0.8rem;
                text-align: left;
                font-weight: 600;
                font-size: 0.85rem;
                white-space: nowrap;
            }}

            th:first-child {{
                border-top-left-radius: 8px;
            }}

            th:last-child {{
                border-top-right-radius: 8px;
            }}

            td {{
                padding: 0.7rem 0.8rem;
                border-bottom: 1px solid #E2E8F0;
                font-size: 0.85rem;
            }}

            tr:last-child td {{
                border-bottom: none;
            }}

            tr:hover {{
                background-color: rgba(102, 126, 234, 0.05);
            }}

            .cluster-label {{
                font-weight: 600;
                color: #1E293B;
            }}

            .metric-selector {{
                margin-bottom: 1rem;
            }}

            select {{
                padding: 0.6rem 1rem;
                border: 1px solid #E2E8F0;
                border-radius: 8px;
                font-size: 0.9rem;
                background-color: white;
                cursor: pointer;
                width: 100%;
                max-width: 300px;
                transition: border-color 0.2s ease;
            }}

            select:focus {{
                outline: none;
                border-color: #3B82F6;
            }}

            label {{
                display: block;
                margin-bottom: 0.5rem;
                color: #64748B;
                font-size: 0.9rem;
                font-weight: 500;
            }}

            .chart-container {{
                height: 400px;
                width: 100%;
            }}

            @media (max-width: 768px) {{
                .two-column {{
                    grid-template-columns: 1fr;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <!-- Header Card -->
            <div class="card">
                <h3>📈 Analisis Mendalam</h3>
            </div>

            <!-- Main Content -->
            <div class="two-column">
                <!-- Left: Cluster Centers Table -->
                <div class="card">
                    <h3>Cluster Centers (Centroid)</h3>
                    <p style="color: #64748B; font-size: 0.85rem; margin-bottom: 1rem;">
                        Nilai rata-rata untuk setiap feature di masing-masing cluster
                    </p>
                    <div class="table-container">
                        <table id="centroidTable">
                            <thead>
                                <tr id="tableHeader">
                                    <!-- Will be populated by JavaScript -->
                                </tr>
                            </thead>
                            <tbody id="centroidBody">
                                <!-- Will be populated by JavaScript -->
                            </tbody>
                        </table>
                    </div>
                </div>

                <!-- Right: Box Plot -->
                <div class="card">
                    <h3>Metrik Trend per Cluster</h3>
                    <p style="color: #64748B; font-size: 0.85rem; margin-bottom: 1rem;">
                        Distribusi nilai metrik dengan deteksi outliers
                    </p>
                    <div class="metric-selector">
                        <label for="metricSelect">Pilih metrik:</label>
                        <select id="metricSelect" onchange="updateBoxPlot()">
                            <!-- Will be populated by JavaScript -->
                        </select>
                    </div>
                    <div id="boxPlot" class="chart-container"></div>
                </div>
            </div>
        </div>

        <script>
            // Data from Python
            const clusterCenters = {centers_json};
            const rawData = {raw_data_json};
            const features = {features_json};

            // Populate table header
            function populateTableHeader() {{
                const thead = document.getElementById('tableHeader');
                let headerHTML = '<th>Cluster</th>';
                
                features.forEach(feature => {{
                    headerHTML += `<th>${{feature}}</th>`;
                }});
                
                thead.innerHTML = headerHTML;
            }}

            // Populate centroid table
            function populateCentroidTable() {{
                const tbody = document.getElementById('centroidBody');
                tbody.innerHTML = '';

                clusterCenters.forEach(center => {{
                    let row = `<tr><td class="cluster-label">Cluster ${{center.cluster}}</td>`;
                    
                    features.forEach(feature => {{
                        const value = center[feature];
                        const formattedValue = value >= 1 ? value.toLocaleString('id-ID', {{maximumFractionDigits: 2}}) : value.toFixed(4);
                        row += `<td>${{formattedValue}}</td>`;
                    }});
                    
                    row += '</tr>';
                    tbody.innerHTML += row;
                }});
            }}

            // Populate metric selector
            function populateMetricSelector() {{
                const select = document.getElementById('metricSelect');
                select.innerHTML = '';
                
                features.forEach(feature => {{
                    const option = document.createElement('option');
                    option.value = feature;
                    option.textContent = feature;
                    select.appendChild(option);
                }});
            }}

            // Create box plot
            function updateBoxPlot() {{
                const metric = document.getElementById('metricSelect').value;
                const traces = [];
                const colors = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6', '#EC4899', '#14B8A6', '#F97316'];

                Object.keys(rawData).sort().forEach((clusterKey, index) => {{
                    const clusterNum = clusterKey.split('_')[1];
                    const data = rawData[clusterKey][metric];
                    
                    if (data && data.length > 0) {{
                        traces.push({{
                            y: data,
                            type: 'box',
                            name: `Cluster ${{clusterNum}}`,
                            marker: {{ color: colors[index % colors.length] }},
                            boxpoints: 'outliers',
                            boxmean: true
                        }});
                    }}
                }});

                const layout = {{
                    height: 400,
                    xaxis: {{ 
                        title: 'Cluster',
                        showgrid: false
                    }},
                    yaxis: {{ 
                        title: metric,
                        showgrid: true,
                        gridcolor: 'rgba(0,0,0,0.05)'
                    }},
                    showlegend: false,
                    margin: {{ t: 30, b: 50, l: 60, r: 30 }},
                    plot_bgcolor: 'rgba(0,0,0,0)',
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    font: {{ 
                        family: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto',
                        size: 11
                    }}
                }};

                const config = {{
                    responsive: true,
                    displayModeBar: false
                }};

                Plotly.newPlot('boxPlot', traces, layout, config);
            }}

            // Initialize
            document.addEventListener('DOMContentLoaded', function() {{
                populateTableHeader();
                populateCentroidTable();
                populateMetricSelector();
                updateBoxPlot();
            }});

            // Handle window resize
            window.addEventListener('resize', function() {{
                if (document.getElementById('boxPlot')) {{
                    Plotly.Plots.resize('boxPlot');
                }}
            }});
        </script>
    </body>
    </html>
    """
    
    # ==================== RENDER HTML COMPONENT ====================
    components.html(html_content, height=850, scrolling=True)