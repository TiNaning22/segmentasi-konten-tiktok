import streamlit.components.v1 as components
import json

def render(df_clustered, result, k_value, features_cols):
    """Render Data tab - Full Hybrid Approach (JavaScript-based)"""
    
    # ==================== PREPARE DATA ====================
    # Define additional columns (metadata)
    metadata_cols = ['contentType', 'type', 'ageGroup', 'location']
    
    # Check which metadata columns exist in the dataframe
    existing_metadata = [col for col in metadata_cols if col in df_clustered.columns]
    
    # Convert DataFrame to list of dicts for JSON
    data_records = df_clustered.to_dict('records')
    
    # Get cluster statistics
    cluster_stats = {}
    for cluster_num in range(k_value):
        cluster_df = df_clustered[df_clustered['Cluster'] == cluster_num]
        cluster_stats[cluster_num] = {
            'count': len(cluster_df),
            'percentage': (len(cluster_df) / len(df_clustered)) * 100
        }
    
    # Prepare color schemes
    color_schemes = {
        'Set3': ['#8DD3C7', '#FFFFB3', '#BEBADA', '#FB8072', '#80B1D3', '#FDB462', '#B3DE69', '#FCCDE5'],
        'Pastel': ['#FBB4AE', '#B3CDE3', '#CCEBC5', '#DECBE4', '#FED9A6', '#FFFFCC', '#E5D8BD', '#FDDAEC'],
        'Plotly': ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880'],
        'D3': ['#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD', '#8C564B', '#E377C2', '#7F7F7F'],
        'Viridis': ['#440154', '#482878', '#3E4A89', '#31688E', '#26828E', '#1F9E89', '#35B779', '#6DCD59'],
        'Safe': ['#88CCEE', '#CC6677', '#DDCC77', '#117733', '#332288', '#AA4499', '#44AA99', '#999933']
    }
    
    # Convert to JSON
    data_json = json.dumps(data_records)
    stats_json = json.dumps(cluster_stats)
    features_json = json.dumps(features_cols)
    metadata_json = json.dumps(existing_metadata)
    colors_json = json.dumps(color_schemes)
    
    # ==================== HTML COMPONENT ====================
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}

            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
                background-color: transparent;
                padding: 1rem;
                color: #1E293B;
            }}

            .container {{
                max-width: 100%;
                margin: 0 auto;
            }}

            .main-grid {{
                display: grid;
                grid-template-columns: 3fr 1fr;
                gap: 1.5rem;
                margin-bottom: 1.5rem;
            }}

            .card {{
                background-color: #FFFFFF;
                border-radius: 12px;
                padding: 1.5rem;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
                border: 1px solid #E2E8F0;
            }}

            .card-header {{
                display: flex;
                align-items: center;
                justify-content: space-between;
                margin-bottom: 1.5rem;
            }}

            h3 {{
                color: #1E293B;
                font-size: 1.2rem;
                font-weight: 600;
                margin: 0;
            }}

            .filter-section {{
                background: linear-gradient(135deg, #F8FAFC 0%, #F1F5F9 100%);
                padding: 1rem 1.2rem;
                border-radius: 10px;
                border: 1px solid #E2E8F0;
                margin-bottom: 1rem;
            }}

            .filter-label {{
                margin: 0 0 0.5rem 0;
                color: #475569;
                font-weight: 600;
                font-size: 0.9rem;
            }}

            .controls-grid {{
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 1rem;
                margin-bottom: 1rem;
            }}

            .control-group {{
                display: flex;
                flex-direction: column;
            }}

            label {{
                display: block;
                margin-bottom: 0.5rem;
                color: #64748B;
                font-size: 0.9rem;
                font-weight: 500;
            }}

            select, input[type="number"] {{
                padding: 0.6rem 1rem;
                border: 1px solid #E2E8F0;
                border-radius: 8px;
                font-size: 0.9rem;
                background-color: white;
                cursor: pointer;
                width: 100%;
                transition: border-color 0.2s ease;
            }}

            select:focus, input[type="number"]:focus {{
                outline: none;
                border-color: #3B82F6;
            }}

            .info-banner {{
                background: linear-gradient(135deg, #F0F9FF 0%, #E0F2FE 100%);
                padding: 0.8rem 1.2rem;
                border-radius: 8px;
                border-left: 4px solid #0EA5E9;
                margin: 1rem 0;
            }}

            .info-banner p {{
                margin: 0;
                color: #0C4A6E;
                font-size: 0.9rem;
            }}

            .table-container {{
                background: white;
                border-radius: 12px;
                overflow: hidden;
                box-shadow: 0 2px 8px rgba(0,0,0,0.04);
                border: 1px solid #E2E8F0;
            }}

            table {{
                width: 100%;
                border-collapse: collapse;
                font-size: 0.875rem;
            }}

            thead {{
                position: sticky;
                top: 0;
                z-index: 10;
            }}

            th {{
                background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%);
                color: white;
                padding: 12px 10px;
                text-align: left;
                font-weight: 600;
                font-size: 0.85rem;
                white-space: nowrap;
            }}

            tbody tr {{
                transition: all 0.2s ease;
            }}

            tbody tr:nth-child(even) {{
                background-color: #F8FAFC;
            }}

            tbody tr:hover {{
                background-color: #EEF2FF;
                transform: scale(1.001);
            }}

            td {{
                padding: 10px;
                border-bottom: 1px solid #E2E8F0;
                color: #1E293B;
            }}

            td:last-child {{
                font-weight: 600;
                color: #3B82F6;
            }}

            /* Metadata columns styling */
            td.metadata-col {{
                background-color: #F8FAFC;
                font-size: 0.8rem;
                color: #64748B;
            }}

            tbody tr:hover td.metadata-col {{
                background-color: #E0E7FF;
            }}

            .table-wrapper {{
                max-height: 500px;
                overflow-y: auto;
            }}

            /* Scrollbar styling */
            .table-wrapper::-webkit-scrollbar {{
                width: 8px;
            }}

            .table-wrapper::-webkit-scrollbar-track {{
                background: #F1F5F9;
                border-radius: 10px;
            }}

            .table-wrapper::-webkit-scrollbar-thumb {{
                background: #CBD5E1;
                border-radius: 10px;
            }}

            .table-wrapper::-webkit-scrollbar-thumb:hover {{
                background: #94A3B8;
            }}

            .download-section {{
                display: grid;
                grid-template-columns: 2fr 1fr;
                gap: 1rem;
                margin-top: 1rem;
            }}

            .btn {{
                padding: 0.75rem 1.5rem;
                border: none;
                border-radius: 8px;
                font-size: 0.9rem;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
                width: 100%;
            }}

            .btn-primary {{
                background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%);
                color: white;
                box-shadow: 0 2px 8px rgba(59, 130, 246, 0.3);
            }}

            .btn-primary:hover {{
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
            }}

            .btn-secondary {{
                background: linear-gradient(135deg, #64748B 0%, #475569 100%);
                color: white;
            }}

            .btn-secondary:hover {{
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(100, 116, 139, 0.3);
            }}

            .sidebar {{
                display: flex;
                flex-direction: column;
                gap: 1rem;
            }}

            .color-scheme-section {{
                background: linear-gradient(135deg, #F8FAFC 0%, #F1F5F9 100%);
                padding: 1rem;
                border-radius: 10px;
                border: 1px solid #E2E8F0;
            }}

            .section-title {{
                margin: 0 0 0.8rem 0;
                color: #475569;
                font-weight: 600;
                font-size: 0.9rem;
            }}

            .color-swatch {{
                padding: 0.8rem 1rem;
                border-radius: 8px;
                margin: 0.5rem 0;
                box-shadow: 0 2px 6px rgba(0,0,0,0.1);
                transition: all 0.3s ease;
                cursor: pointer;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}

            .color-swatch:hover {{
                transform: translateX(5px);
            }}

            .color-swatch-text {{
                color: white;
                font-weight: 600;
                font-size: 0.95rem;
                text-shadow: 0 1px 2px rgba(0,0,0,0.3);
            }}

            .color-swatch-count {{
                color: white;
                font-size: 0.85rem;
                text-shadow: 0 1px 2px rgba(0,0,0,0.3);
            }}

            .badge {{
                margin-left: 0.5rem;
                background: rgba(255,255,255,0.3);
                padding: 0.1rem 0.4rem;
                border-radius: 4px;
                font-size: 0.7rem;
                color: white;
            }}

            .tip-box {{
                background: linear-gradient(135deg, #FEF3C7 0%, #FDE68A 100%);
                padding: 1rem 1.2rem;
                border-radius: 10px;
                border-left: 4px solid #F59E0B;
            }}

            .tip-title {{
                margin: 0 0 0.3rem 0;
                color: #78350F;
                font-weight: 600;
                font-size: 0.9rem;
            }}

            .tip-text {{
                margin: 0;
                color: #92400E;
                font-size: 0.85rem;
                line-height: 1.5;
            }}

            @media (max-width: 768px) {{
                .main-grid {{
                    grid-template-columns: 1fr;
                }}
                
                .controls-grid {{
                    grid-template-columns: 1fr;
                }}
                
                .download-section {{
                    grid-template-columns: 1fr;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="main-grid">
                <!-- Left Column: Data Table -->
                <div>
                    <div class="card">
                        <div class="card-header">
                            <h3>📊 Data dengan Label Cluster</h3>
                        </div>

                        <!-- Filter Section -->
                        <div class="filter-section">
                            <p class="filter-label">⚙️ FILTER & SORT OPTIONS</p>
                        </div>

                        <!-- Controls -->
                        <div class="controls-grid">
                            <div class="control-group">
                                <label for="clusterFilter">🎯 Filter Cluster</label>
                                <select id="clusterFilter" multiple size="5">
                                    <!-- Populated by JavaScript -->
                                </select>
                            </div>

                            <div class="control-group">
                                <label for="sortBy">📊 Urutkan berdasarkan</label>
                                <select id="sortBy" onchange="updateTable()">
                                    <!-- Populated by JavaScript -->
                                </select>
                            </div>

                            <div class="control-group">
                                <label for="displayLimit">📋 Tampilkan rows</label>
                                <input type="number" id="displayLimit" min="10" max="1000" value="100" step="10" onchange="updateTable()">
                            </div>
                        </div>

                        <!-- Info Banner -->
                        <div class="info-banner">
                            <p id="infoBanner">Loading data...</p>
                        </div>

                        <!-- Table -->
                        <div class="table-container">
                            <div class="table-wrapper">
                                <table id="dataTable">
                                    <thead>
                                        <tr id="tableHeader">
                                            <!-- Populated by JavaScript -->
                                        </tr>
                                    </thead>
                                    <tbody id="tableBody">
                                        <!-- Populated by JavaScript -->
                                    </tbody>
                                </table>
                            </div>
                        </div>

                        <!-- Download Section -->
                        <div class="download-section">
                            <button class="btn btn-primary" onclick="downloadFullData()">
                                📥 Download Full Dataset (K={k_value})
                            </button>
                            <button class="btn btn-secondary" onclick="downloadFilteredData()">
                                📥 Download Filtered
                            </button>
                        </div>
                    </div>
                </div>

                <!-- Right Column: Settings & Colors -->
                <div class="sidebar">
                    <div class="card">
                        <h3>⚙️ Display Settings</h3>
                        
                        <div style="margin: 1rem 0;">
                            <label for="colorScheme">🎨 Color Scheme</label>
                            <select id="colorScheme" onchange="updateColors()">
                                <option value="Set3" selected>Set3</option>
                                <option value="Pastel">Pastel</option>
                                <option value="Plotly">Plotly</option>
                                <option value="D3">D3</option>
                                <option value="Viridis">Viridis</option>
                                <option value="Safe">Safe</option>
                            </select>
                        </div>
                    </div>

                    <div class="card">
                        <div class="color-scheme-section">
                            <p class="section-title">🎨 CLUSTER COLORS</p>
                            <div id="colorSwatches">
                                <!-- Populated by JavaScript -->
                            </div>
                        </div>
                    </div>

                    <div class="card">
                        <div class="tip-box">
                            <p class="tip-title">💡 Quick Tip</p>
                            <p class="tip-text">
                                Use filters to focus on specific clusters. Download filtered data for further analysis.
                            </p>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            // Data from Python
            const allData = {data_json};
            const clusterStats = {stats_json};
            const features = {features_json};
            const metadataCols = {metadata_json};
            const colorSchemes = {colors_json};
            const kValue = {k_value};

            let filteredData = [...allData];
            let selectedClusters = Array.from({{length: kValue}}, (_, i) => i);

            // Initialize
            document.addEventListener('DOMContentLoaded', function() {{
                populateClusterFilter();
                populateSortBy();
                populateTableHeader();
                updateColors();
                updateTable();
            }});

            // Populate cluster filter
            function populateClusterFilter() {{
                const select = document.getElementById('clusterFilter');
                select.innerHTML = '';
                
                for (let i = 0; i < kValue; i++) {{
                    const option = document.createElement('option');
                    option.value = i;
                    option.textContent = `Cluster ${{i}}`;
                    option.selected = true;
                    select.appendChild(option);
                }}

                select.addEventListener('change', function() {{
                    selectedClusters = Array.from(this.selectedOptions).map(opt => parseInt(opt.value));
                    updateTable();
                    updateColors();
                }});
            }}

            // Populate sort by dropdown
            function populateSortBy() {{
                const select = document.getElementById('sortBy');
                select.innerHTML = '';
                
                features.forEach(feature => {{
                    const option = document.createElement('option');
                    option.value = feature;
                    option.textContent = feature;
                    select.appendChild(option);
                }});
            }}

            // Populate table header
            function populateTableHeader() {{
                const thead = document.getElementById('tableHeader');
                let headerHTML = '';
                
                // Add metadata columns first
                metadataCols.forEach(col => {{
                    headerHTML += `<th>${{col}}</th>`;
                }});
                
                // Add feature columns
                features.forEach(feature => {{
                    headerHTML += `<th>${{feature}}</th>`;
                }});
                
                // Add cluster column
                headerHTML += '<th>Cluster</th>';
                
                thead.innerHTML = headerHTML;
            }}

            // Update table
            function updateTable() {{
                const sortBy = document.getElementById('sortBy').value;
                const limit = parseInt(document.getElementById('displayLimit').value);

                // Filter data
                filteredData = allData.filter(row => selectedClusters.includes(row.Cluster));

                // Sort data
                filteredData.sort((a, b) => b[sortBy] - a[sortBy]);

                // Limit data
                const displayData = filteredData.slice(0, limit);

                // Update info banner
                document.getElementById('infoBanner').innerHTML = 
                    `<strong>Showing ${{displayData.length.toLocaleString()}}</strong> records from ` +
                    `<strong>${{selectedClusters.length}}</strong> cluster(s) | ` +
                    `Sorted by <strong>${{sortBy}}</strong> (descending)`;

                // Populate table body
                const tbody = document.getElementById('tableBody');
                tbody.innerHTML = '';

                displayData.forEach(row => {{
                    let tr = '<tr>';
                    
                    // Add metadata columns
                    metadataCols.forEach(col => {{
                        const value = row[col] || '-';
                        tr += `<td class="metadata-col">${{value}}</td>`;
                    }});
                    
                    // Add feature columns
                    features.forEach(feature => {{
                        const value = row[feature];
                        const formatted = typeof value === 'number' ? 
                            (value >= 1 ? value.toLocaleString('id-ID', {{maximumFractionDigits: 2}}) : value.toFixed(4)) 
                            : value;
                        tr += `<td>${{formatted}}</td>`;
                    }});
                    
                    // Add cluster column
                    tr += `<td>Cluster ${{row.Cluster}}</td>`;
                    tr += '</tr>';
                    tbody.innerHTML += tr;
                }});
            }}

            // Update colors
            function updateColors() {{
                const scheme = document.getElementById('colorScheme').value;
                const colors = colorSchemes[scheme];
                const container = document.getElementById('colorSwatches');
                container.innerHTML = '';

                for (let i = 0; i < kValue; i++) {{
                    const color = colors[i % colors.length];
                    const stats = clusterStats[i];
                    const isSelected = selectedClusters.includes(i);
                    const opacity = isSelected ? 1 : 0.6;

                    const swatch = document.createElement('div');
                    swatch.className = 'color-swatch';
                    swatch.style.background = `linear-gradient(90deg, ${{color}} 0%, ${{color}}DD 100%)`;
                    swatch.style.opacity = opacity;
                    swatch.style.border = isSelected ? `2px solid ${{color}}` : '2px solid transparent';

                    swatch.innerHTML = `
                        <div>
                            <span class="color-swatch-text">Cluster ${{i}}</span>
                            ${{isSelected ? '<span class="badge">✓ Selected</span>' : ''}}
                        </div>
                        <span class="color-swatch-count">
                            ${{stats.count.toLocaleString()}} (${{stats.percentage.toFixed(1)}}%)
                        </span>
                    `;

                    container.appendChild(swatch);
                }}
            }}

            // Download full data
            function downloadFullData() {{
                downloadCSV(allData, `tiktok_clustered_k${{kValue}}.csv`);
            }}

            // Download filtered data
            function downloadFilteredData() {{
                downloadCSV(filteredData, `tiktok_filtered_k${{kValue}}.csv`);
            }}

            // Helper: Download as CSV
            function downloadCSV(data, filename) {{
                if (data.length === 0) {{
                    alert('No data to download');
                    return;
                }}

                // Create CSV content with metadata + features + cluster
                const headers = [...metadataCols, ...features, 'Cluster'];
                let csv = headers.join(',') + '\\n';

                data.forEach(row => {{
                    const values = headers.map(header => {{
                        const value = row[header];
                        // Handle null/undefined
                        if (value === null || value === undefined) return '';
                        // Quote strings that contain commas
                        return typeof value === 'string' && value.includes(',') ? `"${{value}}"` : value;
                    }});
                    csv += values.join(',') + '\\n';
                }});

                // Create download link
                const blob = new Blob([csv], {{ type: 'text/csv;charset=utf-8;' }});
                const link = document.createElement('a');
                const url = URL.createObjectURL(blob);
                
                link.setAttribute('href', url);
                link.setAttribute('download', filename);
                link.style.visibility = 'hidden';
                
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
            }}
        </script>
    </body>
    </html>
    """
    
    # ==================== RENDER HTML COMPONENT ====================
    components.html(html_content, height=900, scrolling=True)