import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import json

def json_safe(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def get_cluster_insights(df, cluster_col, features_cols):
    """Generate insights per cluster"""
    insights = []
    
    cluster_means = df.groupby(cluster_col)[features_cols].mean()
    cluster_counts = df[cluster_col].value_counts()
    
    engagement_rates = []
    views_list = []
    
    for cluster_num in sorted(df[cluster_col].unique()):
        avg_metrics = cluster_means.loc[cluster_num]
        engagement_rate = avg_metrics.get('Engagement_Rate', 
            (avg_metrics['Likes'] + avg_metrics['Comments'] + avg_metrics['Shares']) / avg_metrics['Views'] if avg_metrics['Views'] > 0 else 0
        )
        engagement_rates.append(engagement_rate)
        views_list.append(avg_metrics['Views'])
    
    engagement_percentiles = pd.Series(engagement_rates).rank(pct=True)
    views_percentiles = pd.Series(views_list).rank(pct=True)
    
    for idx, cluster_num in enumerate(sorted(df[cluster_col].unique())):
        avg_metrics = cluster_means.loc[cluster_num]
        engagement_pct = engagement_percentiles.iloc[idx]
        views_pct = views_percentiles.iloc[idx]
        
        if engagement_pct >= 0.75 and views_pct >= 0.75:
            category = "Golden Content"
            color = "#10B981"
            description = "Viral + Engagement Tinggi"
            emoji = "🌟"
        elif views_pct >= 0.75:
            category = "Passive Viral"
            color = "#3B82F6"
            description = "Views Tinggi, Interaksi Rendah"
            emoji = "👀"
        elif engagement_pct >= 0.75:
            category = "Conversation Starter"
            color = "#F59E0B"
            description = "Engagement Tinggi, Reach Sedang"
            emoji = "💬"
        elif engagement_pct >= 0.50 and views_pct >= 0.50:
            category = "Moderate Performer"
            color = "#6B7280"
            description = "Performa Rata-rata"
            emoji = "📊"
        else:
            category = "Underperforming"
            color = "#EF4444"
            description = "Perlu Optimasi"
            emoji = "⚠️"
        
        insights.append({
            'cluster': cluster_num,
            'category': category,
            'description': description,
            'color': color,
            'emoji': emoji,
            'count': int(cluster_counts[cluster_num]),
            'percentage': float(cluster_counts[cluster_num] / len(df) * 100),
            'avg_likes': float(avg_metrics['Likes']),
            'avg_views': float(avg_metrics['Views']),
            'avg_comments': float(avg_metrics['Comments']),
            'avg_shares': float(avg_metrics['Shares']),
            'avg_engagement': float(avg_metrics['Engagement_Rate']),
            'engagement_rank': float(engagement_pct),
            'views_rank': float(views_pct)
        })
    
    return insights

def get_content_type_distribution(df):
    """Generate ContentType distribution analysis"""
    # Check if ContentType column exists
    if 'ContentType' not in df.columns:
        return None
    
    content_types = []
    
    for content_type in df['ContentType'].unique():
        ct_data = df[df['ContentType'] == content_type]
        
        # Get cluster distribution for this content type
        cluster_dist = ct_data['Cluster'].value_counts()
        dominant_cluster = int(cluster_dist.idxmax())
        dominant_pct = float(cluster_dist.max() / len(ct_data) * 100)
        
        # Calculate averages
        avg_likes = float(ct_data['Likes'].mean())
        avg_views = float(ct_data['Views'].mean())
        avg_comments = float(ct_data['Comments'].mean())
        avg_shares = float(ct_data['Shares'].mean())
        
        # Get age group distribution if available
        age_group_info = "N/A"
        if 'AgeGroup' in df.columns:
            top_age = ct_data['AgeGroup'].mode()
            if len(top_age) > 0:
                age_group_info = str(top_age.iloc[0])
        
        # Determine performance level
        engagement_rate = (avg_likes + avg_comments + avg_shares) / avg_views if avg_views > 0 else 0
        
        if engagement_rate >= 0.05 and avg_views >= df['Views'].median():
            performance = "High"
            perf_color = "#10B981"
        elif engagement_rate >= 0.02 or avg_views >= df['Views'].median():
            performance = "Medium"
            perf_color = "#F59E0B"
        else:
            performance = "Low"
            perf_color = "#EF4444"
        
        content_types.append({
            'content_type': str(content_type),
            'count': int(len(ct_data)),
            'percentage': float(len(ct_data) / len(df) * 100),
            'avg_likes': avg_likes,
            'avg_views': avg_views,
            'avg_comments': avg_comments,
            'avg_shares': avg_shares,
            'engagement_rate': float(engagement_rate),
            'dominant_cluster': dominant_cluster,
            'dominant_cluster_pct': dominant_pct,
            'top_age_group': age_group_info,
            'performance': performance,
            'performance_color': perf_color
        })
    
    # Sort by count descending
    content_types.sort(key=lambda x: x['count'], reverse=True)
    
    return content_types

def render(df_clustered, result, k_value, features_cols):
    """Render Overview tab with ContentType Distribution"""
    
    # Prepare all data
    insights = get_cluster_insights(df_clustered, 'Cluster', features_cols)
    
    cluster_counts = df_clustered['Cluster'].value_counts().sort_index()
    distribution_data = {
        'labels': [f"Cluster {i}" for i in cluster_counts.index],
        'values': cluster_counts.values.tolist()
    }
    
    cluster_means = df_clustered.groupby('Cluster')[features_cols].mean()
    main_features = ['Likes', 'Views', 'Comments', 'Shares'] if all(f in features_cols for f in ['Likes', 'Views', 'Comments', 'Shares']) else features_cols[:4]
    
    bar_chart_data = {
        'clusters': [f"Cluster {i}" for i in cluster_means.index],
        'metrics': {feature: cluster_means[feature].tolist() for feature in main_features}
    }
    
    # Get ContentType distribution
    content_type_data = get_content_type_distribution(df_clustered)
    has_content_type = content_type_data is not None
    
    # Convert to JSON
    insights_json = json.dumps(insights, default=json_safe)
    distribution_json = json.dumps(distribution_data, default=json_safe)
    bar_chart_json = json.dumps(bar_chart_data, default=json_safe)
    content_type_json = json.dumps(content_type_data, default=json_safe) if has_content_type else json.dumps(None)
    has_content_type_json = json.dumps(has_content_type)
    
    # HTML Component (includes ContentType table rendering)
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.27.0/plotly.min.js"></script>
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background-color: transparent; color: #1E293B; }}
            .card {{ background-color: #FFFFFF; border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem; box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05); }}
            .section-header {{ font-size: 1.2rem; font-weight: 600; color: #1E293B; margin-bottom: 1rem; }}
            .section-subtitle {{ font-size: 0.9rem; color: #64748B; margin-bottom: 1rem; }}
            .two-column {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-bottom: 1.5rem; }}
            .chart-container {{ height: 420px; width: 100%; }}
            .insight-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem; margin-bottom: 1.5rem; }}
            .insight-card {{ border-radius: 12px; padding: 1.5rem; border-left: 5px solid; box-shadow: 0 2px 8px rgba(0,0,0,0.08); transition: transform 0.2s; }}
            .insight-card:hover {{ transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.12); }}
            .insight-header {{ display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.8rem; }}
            .insight-cluster {{ font-size: 1.1rem; font-weight: 600; }}
            .insight-emoji {{ font-size: 1.5rem; }}
            .summary-table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; margin-top: 1rem; }}
            .summary-table th {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 0.8rem; text-align: left; font-weight: 600; }}
            .summary-table th:first-child {{ border-top-left-radius: 8px; }}
            .summary-table th:last-child {{ border-top-right-radius: 8px; }}
            .summary-table td {{ padding: 1rem 0.8rem; border-bottom: 1px solid #E2E8F0; vertical-align: middle; }}
            .summary-table tr:hover {{ background-color: rgba(102, 126, 234, 0.05); }}
            .mini-bar {{ height: 4px; border-radius: 2px; margin-top: 4px; }}
            .cluster-badge {{ display: inline-block; padding: 4px 10px; background-color: rgba(59, 130, 246, 0.15); color: #3B82F6; border-radius: 6px; font-weight: 600; font-size: 0.85rem; }}
            .performance-badge {{ display: inline-block; padding: 4px 10px; border-radius: 6px; font-size: 0.75rem; font-weight: 600; }}
            @media (max-width: 768px) {{ .two-column, .insight-grid {{ grid-template-columns: 1fr; }} }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="card"><h3 class="section-header">📊 Distribusi & Performa Cluster</h3></div>
            
            <div class="two-column">
                <div class="card">
                    <h4 style="font-size: 1rem; font-weight: 600; margin-bottom: 1rem;">Distribusi Konten per Cluster</h4>
                    <div id="pieChart" class="chart-container"></div>
                </div>
                <div class="card">
                    <h4 style="font-size: 1rem; font-weight: 600; margin-bottom: 1rem;">Rata-rata Metrik per Cluster</h4>
                    <div id="barChart" class="chart-container"></div>
                </div>
            </div>

            <div class="card">
                <h3 class="section-header">🎯 Profil Cluster</h3>
                <p class="section-subtitle">Interpretasi otomatis berdasarkan engagement dan views</p>
            </div>
            <div id="insightCards" class="insight-grid"></div>

            <div class="card">
                <h3 class="section-header">📈 Ringkasan Metrik Detail</h3>
                <div style="overflow-x: auto;"><table class="summary-table" id="summaryTable"></table></div>
            </div>

            <div id="contentTypeSection" style="display: none;">
                <div class="card">
                    <h3 class="section-header">🎬 Distribusi Content Type</h3>
                    <p class="section-subtitle">Performa berdasarkan jenis konten dengan cluster dominan dan age group</p>
                    <div style="overflow-x: auto;"><table class="summary-table" id="contentTypeTable"></table></div>
                </div>
            </div>
        </div>

        <script>
            const insights = {insights_json};
            const distributionData = {distribution_json};
            const barChartData = {bar_chart_json};
            const contentTypeData = {content_type_json};
            const hasContentType = {has_content_type_json};

            function createPieChart() {{
                Plotly.newPlot('pieChart', [{{
                    values: distributionData.values,
                    labels: distributionData.labels,
                    type: 'pie',
                    hole: 0.4,
                    marker: {{ colors: ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6', '#EC4899'], line: {{ color: '#FFF', width: 2 }} }},
                    textinfo: 'percent+label'
                }}], {{ height: 420, showlegend: true, margin: {{ t: 20, b: 20, l: 20, r: 100 }}, paper_bgcolor: 'rgba(0,0,0,0)' }}, {{ responsive: true, displayModeBar: false }});
            }}

            function createBarChart() {{
                const traces = Object.keys(barChartData.metrics).map((metric, idx) => ({{
                    x: barChartData.clusters,
                    y: barChartData.metrics[metric],
                    name: metric,
                    type: 'bar',
                    marker: {{ color: ['#3B82F6', '#10B981', '#F59E0B', '#EF4444'][idx] }}
                }}));
                Plotly.newPlot('barChart', traces, {{ barmode: 'group', height: 420, yaxis: {{ title: 'Nilai Rata-rata' }}, margin: {{ t: 60, b: 40, l: 60, r: 20 }}, paper_bgcolor: 'rgba(0,0,0,0)' }}, {{ responsive: true, displayModeBar: false }});
            }}

            function createInsightCards() {{
                const html = insights.map(i => `
                    <div class="insight-card" style="background: linear-gradient(135deg, ${{i.color}}08 0%, ${{i.color}}15 100%); border-left-color: ${{i.color}};">
                        <div class="insight-header">
                            <h4 class="insight-cluster">Cluster ${{i.cluster}}</h4>
                            <span class="insight-emoji">${{i.emoji}}</span>
                        </div>
                        <p style="font-weight: 600; color: ${{i.color}};">${{i.category}}</p>
                        <p style="font-size: 0.85rem; color: #64748B;">${{i.description}}</p>
                        <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid ${{i.color}}30;">
                            <p><strong>${{i.count.toLocaleString('id-ID')}}</strong> konten</p>
                            <p style="font-size: 0.8rem; color: #64748B;">${{i.percentage.toFixed(1)}}% dari total</p>
                        </div>
                    </div>
                `).join('');
                document.getElementById('insightCards').innerHTML = html;
            }}

            function createSummaryTable() {{
                let html = '<thead><tr><th>Cluster</th><th>Kategori</th><th>Jumlah</th><th>%</th><th>Likes</th><th>Views</th><th>Comments</th><th>Shares</th><th>Engagement</th></tr></thead><tbody>';
                insights.forEach(i => {{
                    html += `<tr><td><strong>Cluster ${{i.cluster}}</strong></td><td>${{i.category}}</td><td>${{i.count.toLocaleString('id-ID')}}</td><td>${{i.percentage.toFixed(1)}}%</td><td>${{i.avg_likes.toLocaleString('id-ID', {{maximumFractionDigits: 0}})}}</td><td>${{i.avg_views.toLocaleString('id-ID', {{maximumFractionDigits: 0}})}}</td><td>${{i.avg_comments.toLocaleString('id-ID', {{maximumFractionDigits: 0}})}}</td><td>${{i.avg_shares.toLocaleString('id-ID', {{maximumFractionDigits: 0}})}}</td><td>${{i.avg_engagement.toFixed(4)}}</td></tr>`;
                }});
                document.getElementById('summaryTable').innerHTML = html + '</tbody>';
            }}

            function createContentTypeTable() {{
                if (!hasContentType || !contentTypeData) return;
                document.getElementById('contentTypeSection').style.display = 'block';
                
                let html = '<thead><tr><th>Content Type</th><th>Jumlah</th><th>%</th><th>Likes</th><th>Views</th><th>Comments</th><th>Shares</th><th>Age Group</th><th>Cluster Dominan</th><th>Performance</th></tr></thead><tbody>';
                contentTypeData.forEach(ct => {{
                    html += `<tr style="border-left: 3px solid ${{ct.performance_color}};">
                        <td><strong>${{ct.content_type}}</strong><div class="mini-bar" style="width: ${{ct.percentage}}%; background: linear-gradient(90deg, ${{ct.performance_color}} 0%, ${{ct.performance_color}}80 100%);"></div></td>
                        <td style="text-align: right; font-weight: 600;">${{ct.count.toLocaleString('id-ID')}}</td>
                        <td style="text-align: right;">${{ct.percentage.toFixed(1)}}%</td>
                        <td style="text-align: right;">${{ct.avg_likes.toLocaleString('id-ID', {{maximumFractionDigits: 0}})}}</td>
                        <td style="text-align: right;">${{ct.avg_views.toLocaleString('id-ID', {{maximumFractionDigits: 0}})}}</td>
                        <td style="text-align: right;">${{ct.avg_comments.toLocaleString('id-ID', {{maximumFractionDigits: 0}})}}</td>
                        <td style="text-align: right;">${{ct.avg_shares.toLocaleString('id-ID', {{maximumFractionDigits: 0}})}}</td>
                        <td style="text-align: center; color: #64748B;">${{ct.top_age_group}}</td>
                        <td style="text-align: center;"><span class="cluster-badge">C${{ct.dominant_cluster}} <span style="font-size: 0.75rem;">(${{ct.dominant_cluster_pct.toFixed(0)}}%)</span></span></td>
                        <td style="text-align: center;"><span class="performance-badge" style="background: ${{ct.performance_color}}20; color: ${{ct.performance_color}};">${{ct.performance}}</span></td>
                    </tr>`;
                }});
                document.getElementById('contentTypeTable').innerHTML = html + '</tbody>';
            }}

            document.addEventListener('DOMContentLoaded', () => {{
                createPieChart();
                createBarChart();
                createInsightCards();
                createSummaryTable();
                createContentTypeTable();
            }});

            window.addEventListener('resize', () => {{
                Plotly.Plots.resize('pieChart');
                Plotly.Plots.resize('barChart');
            }});
        </script>
    </body>
    </html>
    """
    
    components.html(html_content, height=2400, scrolling=True)