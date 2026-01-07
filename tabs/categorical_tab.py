import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import json

def render(df_clustered, result, k_value, features_cols):
    """Render Categorical Profiling tab - Hybrid Version"""
    
    # ==================== DETECT CATEGORICAL COLUMNS ====================
    categorical_cols = df_clustered.select_dtypes(include=['object', 'category']).columns.tolist()
    exclude_cols = ['Cluster']
    categorical_cols = [col for col in categorical_cols if col not in exclude_cols]
    
    # ==================== IF NO CATEGORICAL COLUMNS - CREATE DEMO ====================
    if len(categorical_cols) == 0:
        st.warning("⚠️ Tidak ada kolom kategorikal yang terdeteksi di dataset.")
        
        st.info("""
        **Tips:** Dataset Anda mungkin hanya berisi data numerik. Untuk profiling kategorikal, 
        pastikan dataset memiliki kolom seperti:
        - `ContentType` (Video, Image, Text)
        - `Demographics` (Age Group, Gender, Location)
        - `Hashtag` atau `Category`
        - `DeviceType` (Mobile, Desktop)
        """)
        
        st.markdown("---")
        st.markdown("#### 🎭 Contoh Profiling Kategorikal (Data Demo)")
        
        # Create demo data
        demo_data = pd.DataFrame({
            'Cluster': df_clustered['Cluster'],
            'ContentType_Demo': pd.cut(df_clustered['Likes'], bins=3, labels=['Low Engagement', 'Medium Engagement', 'High Engagement']),
            'Demographics_Demo': pd.cut(df_clustered['Views'], bins=3, labels=['Young Audience', 'Adult Audience', 'Senior Audience']),
            'Platform_Demo': pd.cut(df_clustered['Engagement_Rate'], bins=3, labels=['Android', 'iOS', 'Web'])
        })
        
        categorical_cols = ['ContentType_Demo', 'Demographics_Demo', 'Platform_Demo']
        df_demo = df_clustered.copy()
        for col in categorical_cols:
            df_demo[col] = demo_data[col].astype(str)
        
        render_categorical_analysis(df_demo, categorical_cols, k_value, is_demo=True)
    else:
        # st.success(f"✅ Ditemukan {len(categorical_cols)} kolom kategorikal: {', '.join(categorical_cols)}")
        render_categorical_analysis(df_clustered, categorical_cols, k_value, is_demo=False)

def render_categorical_analysis(df, categorical_cols, k_value, is_demo=False):
    """Render categorical analysis with HTML component"""
    
    # ==================== PREPARE DATA FOR ALL COLUMNS ====================
    all_categorical_data = {}
    
    for col in categorical_cols:
        # Crosstab percentage
        crosstab_pct = pd.crosstab(df['Cluster'], df[col], normalize='index') * 100
        
        # Crosstab count
        crosstab_count = pd.crosstab(df['Cluster'], df[col])
        
        # Insights per cluster
        insights = []
        for cluster_num in sorted(df['Cluster'].unique()):
            cluster_data = crosstab_pct.loc[cluster_num]
            dominant_category = cluster_data.idxmax()
            dominant_percentage = cluster_data.max()
            diverse_categories = (cluster_data > 10).sum()
            diversity_level = "Tinggi" if diverse_categories >= 3 else "Sedang" if diverse_categories == 2 else "Rendah"
            
            insights.append({
                'cluster': int(cluster_num),
                'dominant': str(dominant_category),
                'percentage': float(dominant_percentage),
                'diversity': diversity_level,
                'diverse_count': int(diverse_categories)
            })
        
        # Summary statistics
        summary_stats = []
        for cat_value in df[col].unique():
            cat_data = df[df[col] == cat_value]
            cluster_dist = cat_data['Cluster'].value_counts(normalize=True) * 100
            
            summary_stats.append({
                'category': str(cat_value),
                'count': int(len(cat_data)),
                'percentage': float(len(cat_data) / len(df) * 100),
                'dominant_cluster': int(cluster_dist.idxmax()),
                'cluster_percentage': float(cluster_dist.max())
            })
        
        # Prepare chart data
        categories = crosstab_pct.columns.tolist()
        clusters = crosstab_pct.index.tolist()
        
        # Bar chart data (percentage)
        bar_data = {
            'clusters': [f"Cluster {i}" for i in clusters],
            'categories': {}
        }
        for cat in categories:
            bar_data['categories'][str(cat)] = crosstab_pct[cat].tolist()
        
        # Heatmap data
        heatmap_data = {
            'z': crosstab_pct.values.tolist(),
            'x': [str(cat) for cat in categories],
            'y': [f"Cluster {i}" for i in clusters]
        }
        
        # Count table data
        count_table = {
            'clusters': [f"Cluster {i}" for i in clusters],
            'data': {}
        }
        for cat in categories:
            count_table['data'][str(cat)] = crosstab_count[cat].tolist()
        
        all_categorical_data[col] = {
            'insights': insights,
            'summary_stats': summary_stats,
            'bar_data': bar_data,
            'heatmap_data': heatmap_data,
            'count_table': count_table,
            'categories': [str(c) for c in categories]
        }
    
    # Convert to JSON
    categorical_json = json.dumps(all_categorical_data)
    cols_json = json.dumps(categorical_cols)
    k_value_json = json.dumps(k_value)
    
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
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background-color: transparent;
                color: #1E293B;
            }}

            .card {{
                background-color: #FFFFFF;
                border-radius: 12px;
                padding: 1.5rem;
                margin-bottom: 1.5rem;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
            }}

            .section-header {{
                font-size: 1.2rem;
                font-weight: 600;
                color: #1E293B;
                margin-bottom: 0.5rem;
            }}

            .section-subtitle {{
                font-size: 0.9rem;
                color: #64748B;
                margin-bottom: 1rem;
            }}

            .selector-container {{
                margin-bottom: 1.5rem;
            }}

            select {{
                width: 100%;
                max-width: 400px;
                padding: 0.7rem 1rem;
                border: 1px solid #E2E8F0;
                border-radius: 8px;
                font-size: 0.9rem;
                background-color: white;
                cursor: pointer;
                transition: border-color 0.2s ease;
            }}

            select:focus {{
                outline: none;
                border-color: #3B82F6;
                box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
            }}

            label {{
                display: block;
                margin-bottom: 0.5rem;
                color: #64748B;
                font-size: 0.9rem;
                font-weight: 500;
            }}

            .two-column {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 1.5rem;
                margin-bottom: 1.5rem;
            }}

            .chart-container {{
                height: 420px;
                width: 100%;
            }}

            .insight-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
                gap: 1rem;
                margin-bottom: 1.5rem;
            }}

            .insight-card {{
                border-radius: 10px;
                padding: 1.2rem;
                border-left: 4px solid;
                box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            }}

            .insight-cluster-label {{
                font-size: 1rem;
                font-weight: 600;
                color: #1E293B;
                margin-bottom: 0.5rem;
            }}

            .insight-dominant {{
                font-size: 0.95rem;
                font-weight: 600;
                margin: 0.5rem 0;
            }}

            .insight-percentage {{
                font-size: 0.85rem;
                color: #475569;
                margin: 0.3rem 0;
            }}

            .insight-diversity {{
                font-size: 0.8rem;
                color: #64748B;
                margin-top: 0.5rem;
                padding-top: 0.5rem;
                border-top: 1px solid rgba(0,0,0,0.1);
            }}

            .summary-table {{
                width: 100%;
                border-collapse: collapse;
                font-size: 0.85rem;
                margin-top: 1rem;
            }}

            .summary-table th {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 0.8rem;
                text-align: left;
                font-weight: 600;
                font-size: 0.85rem;
            }}

            .summary-table th:first-child {{
                border-top-left-radius: 8px;
            }}

            .summary-table th:last-child {{
                border-top-right-radius: 8px;
            }}

            .summary-table td {{
                padding: 0.7rem 0.8rem;
                border-bottom: 1px solid #E2E8F0;
            }}

            .summary-table tr:last-child td {{
                border-bottom: none;
            }}

            .summary-table tr:hover {{
                background-color: rgba(102, 126, 234, 0.05);
            }}

            .tab-container {{
                display: flex;
                gap: 0.5rem;
                margin-bottom: 1rem;
                border-bottom: 2px solid #E2E8F0;
            }}

            .tab-button {{
                padding: 0.7rem 1.5rem;
                border: none;
                background: transparent;
                color: #64748B;
                font-size: 0.9rem;
                cursor: pointer;
                border-bottom: 3px solid transparent;
                transition: all 0.2s ease;
            }}

            .tab-button:hover {{
                color: #3B82F6;
            }}

            .tab-button.active {{
                color: #3B82F6;
                border-bottom-color: #3B82F6;
            }}

            .tab-content {{
                display: none;
            }}

            .tab-content.active {{
                display: block;
            }}

            .recommendation-box {{
                padding: 1rem;
                border-radius: 8px;
                margin-bottom: 0.8rem;
                border-left: 4px solid;
            }}

            .rec-success {{
                background-color: #10B98110;
                border-left-color: #10B981;
            }}

            .rec-info {{
                background-color: #3B82F610;
                border-left-color: #3B82F6;
            }}

            .rec-warning {{
                background-color: #F59E0B10;
                border-left-color: #F59E0B;
            }}

            @media (max-width: 768px) {{
                .two-column {{
                    grid-template-columns: 1fr;
                }}
                
                .insight-grid {{
                    grid-template-columns: 1fr;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <!-- Header -->
           

            <!-- Column Selector -->
            <div class="card selector-container">
                <label for="columnSelect">Pilih Kolom Kategorikal untuk Analisis:</label>
                <select id="columnSelect" onchange="updateAnalysis()">
                    <!-- Will be populated by JavaScript -->
                </select>
            </div>

            <!-- Distribution Section -->
            <div class="card">
                <h3 class="section-header">Distribusi <span id="selectedColumnTitle"></span> per Cluster</h3>
            </div>

            <div class="two-column">
                <!-- Bar Chart -->
                <div class="card">
                    <h4 style="font-size: 1rem; font-weight: 600; margin-bottom: 1rem;">📈 Visualisasi Persentase</h4>
                    <div id="barChart" class="chart-container"></div>
                </div>

                <!-- Heatmap -->
                <div class="card">
                    <h4 style="font-size: 1rem; font-weight: 600; margin-bottom: 1rem;">🎯 Heatmap Distribusi</h4>
                    <div id="heatmap" class="chart-container"></div>
                </div>
            </div>

            <!-- Insights Section -->
            <div class="card">
                <h3 class="section-header">🎯 Insight Otomatis per Cluster</h3>
                <div id="insightCards" class="insight-grid">
                    <!-- Will be populated by JavaScript -->
                </div>
            </div>

            <!-- Tables Section -->
            <div class="card">
                <h3 class="section-header">📋 Tabel Detail</h3>
                <div class="tab-container">
                    <button class="tab-button active" onclick="switchTab('percentage')">Persentase (%)</button>
                    <button class="tab-button" onclick="switchTab('count')">Jumlah Absolut</button>
                </div>
                <div id="percentageTab" class="tab-content active">
                    <div style="overflow-x: auto;">
                        <table class="summary-table" id="percentageTable"></table>
                    </div>
                </div>
                <div id="countTab" class="tab-content">
                    <div style="overflow-x: auto;">
                        <table class="summary-table" id="countTable"></table>
                    </div>
                </div>
            </div>

            <!-- Summary Statistics -->
            <div class="card">
                <h3 class="section-header">📊 Statistik Summary</h3>
                <p class="section-subtitle">Distribusi kategori di seluruh cluster</p>
                <div style="overflow-x: auto;">
                    <table class="summary-table" id="summaryStatsTable"></table>
                </div>
            </div>

            <!-- Recommendations -->
            <div class="card">
                <h3 class="section-header">💡 Rekomendasi Berdasarkan Profiling</h3>
                <div id="recommendations"></div>
            </div>
        </div>

        <script>
            // Data from Python
            const categoricalData = {categorical_json};
            const columnsList = {cols_json};
            const kValue = {k_value_json};
            let currentColumn = null;

            // Initialize
            function init() {{
                populateColumnSelector();
                currentColumn = columnsList[0];
                updateAnalysis();
            }}

            // Populate column selector
            function populateColumnSelector() {{
                const select = document.getElementById('columnSelect');
                select.innerHTML = '';
                
                columnsList.forEach(col => {{
                    const option = document.createElement('option');
                    option.value = col;
                    option.textContent = col;
                    select.appendChild(option);
                }});
            }}

            // Update all visualizations
            function updateAnalysis() {{
                currentColumn = document.getElementById('columnSelect').value;
                const data = categoricalData[currentColumn];
                
                document.getElementById('selectedColumnTitle').textContent = currentColumn;
                
                createBarChart(data);
                createHeatmap(data);
                createInsightCards(data);
                createPercentageTable(data);
                createCountTable(data);
                createSummaryStatsTable(data);
                createRecommendations(data);
            }}

            // Create Bar Chart
            function createBarChart(data) {{
                const traces = [];
                const colors = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6', '#EC4899', '#14B8A6', '#F97316'];
                
                Object.keys(data.bar_data.categories).forEach((category, idx) => {{
                    traces.push({{
                        x: data.bar_data.clusters,
                        y: data.bar_data.categories[category],
                        name: category,
                        type: 'bar',
                        marker: {{ color: colors[idx % colors.length] }}
                    }});
                }});

                const layout = {{
                    barmode: 'group',
                    height: 420,
                    xaxis: {{ title: 'Cluster', showgrid: false }},
                    yaxis: {{ title: 'Persentase (%)', showgrid: true, gridcolor: 'rgba(0,0,0,0.05)' }},
                    legend: {{ orientation: 'h', yanchor: 'bottom', y: 1.02, xanchor: 'right', x: 1 }},
                    margin: {{ t: 60, b: 40, l: 60, r: 20 }},
                    plot_bgcolor: 'rgba(0,0,0,0)',
                    paper_bgcolor: 'rgba(0,0,0,0)'
                }};

                Plotly.newPlot('barChart', traces, layout, {{ responsive: true, displayModeBar: false }});
            }}

            // Create Heatmap
            function createHeatmap(data) {{
                const trace = {{
                    z: data.heatmap_data.z,
                    x: data.heatmap_data.x,
                    y: data.heatmap_data.y,
                    type: 'heatmap',
                    colorscale: 'Blues',
                    hovertemplate: '%{{y}}<br>%{{x}}<br>%{{z:.1f}}%<extra></extra>'
                }};

                const layout = {{
                    height: 420,
                    xaxis: {{ title: currentColumn }},
                    yaxis: {{ title: 'Cluster' }},
                    margin: {{ t: 30, b: 60, l: 80, r: 20 }},
                    paper_bgcolor: 'rgba(0,0,0,0)'
                }};

                Plotly.newPlot('heatmap', [trace], layout, {{ responsive: true, displayModeBar: false }});
            }}

            // Create Insight Cards
            function createInsightCards(data) {{
                const container = document.getElementById('insightCards');
                container.innerHTML = '';

                const colors = ['#10B981', '#F59E0B', '#EF4444'];
                
                data.insights.forEach(insight => {{
                    const colorIdx = insight.percentage > 50 ? 0 : insight.percentage > 30 ? 1 : 2;
                    const color = colors[colorIdx];
                    
                    const card = document.createElement('div');
                    card.className = 'insight-card';
                    card.style.background = `linear-gradient(135deg, ${{color}}10 0%, ${{color}}20 100%)`;
                    card.style.borderLeftColor = color;
                    
                    card.innerHTML = `
                        <div class="insight-cluster-label">Cluster ${{insight.cluster}}</div>
                        <div class="insight-dominant" style="color: ${{color}};">${{insight.dominant}}</div>
                        <div class="insight-percentage"><strong>${{insight.percentage.toFixed(1)}}%</strong> dominan</div>
                        <div class="insight-diversity">Diversitas: ${{insight.diversity}} (${{insight.diverse_count}} kategori)</div>
                    `;
                    
                    container.appendChild(card);
                }});
            }}

            // Create Percentage Table
            function createPercentageTable(data) {{
                const table = document.getElementById('percentageTable');
                let html = '<thead><tr><th>Cluster</th>';
                
                data.categories.forEach(cat => {{
                    html += `<th>${{cat}}</th>`;
                }});
                html += '</tr></thead><tbody>';
                
                data.bar_data.clusters.forEach((cluster, idx) => {{
                    html += `<tr><td><strong>${{cluster}}</strong></td>`;
                    data.categories.forEach(cat => {{
                        const value = data.bar_data.categories[cat][idx];
                        html += `<td>${{value.toFixed(1)}}%</td>`;
                    }});
                    html += '</tr>';
                }});
                
                html += '</tbody>';
                table.innerHTML = html;
            }}

            // Create Count Table
            function createCountTable(data) {{
                const table = document.getElementById('countTable');
                let html = '<thead><tr><th>Cluster</th>';
                
                data.categories.forEach(cat => {{
                    html += `<th>${{cat}}</th>`;
                }});
                html += '</tr></thead><tbody>';
                
                data.count_table.clusters.forEach((cluster, idx) => {{
                    html += `<tr><td><strong>${{cluster}}</strong></td>`;
                    data.categories.forEach(cat => {{
                        const value = data.count_table.data[cat][idx];
                        html += `<td>${{value.toLocaleString('id-ID')}}</td>`;
                    }});
                    html += '</tr>';
                }});
                
                html += '</tbody>';
                table.innerHTML = html;
            }}

            // Create Summary Stats Table
            function createSummaryStatsTable(data) {{
                const table = document.getElementById('summaryStatsTable');
                let html = `
                    <thead>
                        <tr>
                            <th>Kategori</th>
                            <th>Jumlah</th>
                            <th>Persentase Total</th>
                            <th>Cluster Dominan</th>
                            <th>% di Cluster Dominan</th>
                        </tr>
                    </thead>
                    <tbody>
                `;
                
                data.summary_stats.forEach(stat => {{
                    html += `
                        <tr>
                            <td><strong>${{stat.category}}</strong></td>
                            <td>${{stat.count.toLocaleString('id-ID')}}</td>
                            <td>${{stat.percentage.toFixed(1)}}%</td>
                            <td>Cluster ${{stat.dominant_cluster}}</td>
                            <td>${{stat.cluster_percentage.toFixed(1)}}%</td>
                        </tr>
                    `;
                }});
                
                html += '</tbody>';
                table.innerHTML = html;
            }}

            // Create Recommendations
            function createRecommendations(data) {{
                const container = document.getElementById('recommendations');
                container.innerHTML = '';
                
                data.insights.forEach(insight => {{
                    let recType, recText;
                    
                    if (insight.percentage > 70) {{
                        recType = 'success';
                        recText = `Cluster ${{insight.cluster}} sangat homogen (${{insight.percentage.toFixed(0)}}% ${{insight.dominant}}). Fokus pada konten untuk segmen ini.`;
                    }} else if (insight.percentage > 40) {{
                        recType = 'info';
                        recText = `Cluster ${{insight.cluster}} didominasi ${{insight.dominant}} (${{insight.percentage.toFixed(0)}}%). Prioritaskan strategi untuk segmen ini.`;
                    }} else {{
                        recType = 'warning';
                        recText = `Cluster ${{insight.cluster}} cukup beragam. Pertimbangkan strategi multi-segment untuk berbagai kategori.`;
                    }}
                    
                    const box = document.createElement('div');
                    box.className = `recommendation-box rec-${{recType}}`;
                    box.innerHTML = `<strong>Cluster ${{insight.cluster}}:</strong> ${{recText}}`;
                    container.appendChild(box);
                }});
            }}

            // Switch tabs
            function switchTab(tabName) {{
                // Update buttons
                document.querySelectorAll('.tab-button').forEach(btn => {{
                    btn.classList.remove('active');
                }});
                event.target.classList.add('active');
                
                // Update content
                document.querySelectorAll('.tab-content').forEach(content => {{
                    content.classList.remove('active');
                }});
                document.getElementById(tabName + 'Tab').classList.add('active');
            }}

            // Initialize on load
            document.addEventListener('DOMContentLoaded', init);
            
            // Handle resize
            window.addEventListener('resize', function() {{
                Plotly.Plots.resize('barChart');
                Plotly.Plots.resize('heatmap');
            }});
        </script>
    </body>
    </html>
    """
    
    # ==================== RENDER HTML COMPONENT ====================
    components.html(html_content, height=2200, scrolling=True)