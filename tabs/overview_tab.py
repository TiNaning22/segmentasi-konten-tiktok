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
    """Generate insights per cluster with unique categories using ranking"""
    insights = []
    
    cluster_means = df.groupby(cluster_col)[features_cols].mean()
    cluster_counts = df[cluster_col].value_counts()
    
    # Calculate engagement rate and views for each cluster
    cluster_metrics = []
    for cluster_num in sorted(df[cluster_col].unique()):
        avg_metrics = cluster_means.loc[cluster_num]
        engagement_rate = avg_metrics.get('Engagement_Rate', 
            (avg_metrics['Likes'] + avg_metrics['Comments'] + avg_metrics['Shares']) / avg_metrics['Views'] if avg_metrics['Views'] > 0 else 0
        )
        
        cluster_metrics.append({
            'cluster': cluster_num,
            'engagement': engagement_rate,
            'views': avg_metrics['Views'],
            'likes': avg_metrics['Likes'],
            'comments': avg_metrics['Comments'],
            'shares': avg_metrics['Shares']
        })
    
    # Sort by engagement and views to get rankings
    df_metrics = pd.DataFrame(cluster_metrics)
    df_metrics['engagement_rank'] = df_metrics['engagement'].rank(ascending=False, method='min')
    df_metrics['views_rank'] = df_metrics['views'].rank(ascending=False, method='min')
    
    # Calculate percentiles for reference
    df_metrics['engagement_pct'] = df_metrics['engagement'].rank(pct=True)
    df_metrics['views_pct'] = df_metrics['views'].rank(pct=True)
    
    # Assign unique categories based on combined ranking
    # Create unique identifier for each cluster
    assigned_categories = set()
    
    for idx, row in df_metrics.iterrows():
        cluster_num = row['cluster']
        engagement_rank = int(row['engagement_rank'])
        views_rank = int(row['views_rank'])
        engagement_pct = row['engagement_pct']
        views_pct = row['views_pct']
        
        # Use combined score for more nuanced categorization
        combined_score = (engagement_rank + views_rank) / 2
        
        # Categorize based on rank combinations (ensures uniqueness)
        # Priority 1: Best performers
        if engagement_rank == 1 and views_rank == 1:
            category = "Golden Content"
            color = "#10B981"
            description = "Top Engagement + Top Views"
            emoji = ""
        
        # Priority 2: Single dimension leaders
        elif views_rank == 1 and engagement_rank > 1:
            category = "Passive Viral"
            color = "#3B82F6"
            description = f"Viral (#1 Views, #{engagement_rank} Engagement)"
            emoji = ""
        elif engagement_rank == 1 and views_rank > 1:
            category = "Conversation Starter"
            color = "#F59E0B"
            description = f"High Engagement (#1), #{views_rank} Views"
            emoji = ""
        
        # Priority 3: Second place holders
        elif views_rank == 2 and engagement_rank != 2:
            category = "High Reach"
            color = "#06B6D4"
            description = f"2nd Highest Views, #{engagement_rank} Engagement"
            emoji = ""
        elif engagement_rank == 2 and views_rank != 2:
            category = "Engaging Content"
            color = "#8B5CF6"
            description = f"2nd Best Engagement, #{views_rank} Views"
            emoji = ""
        
        # Priority 4: Balanced performers
        elif engagement_rank == 2 and views_rank == 2:
            category = "Balanced Runner-up"
            color = "#14B8A6"
            description = f"2nd Place: #{engagement_rank} Eng, #{views_rank} Views"
            emoji = ""
        elif combined_score <= 3.0:
            category = "Strong Performer"
            color = "#10B981"
            description = f"Solid: #{engagement_rank} Eng, #{views_rank} Views"
            emoji = ""
        
        # Priority 5: Middle tier - use views as primary differentiator
        elif views_rank == 3:
            category = "Growing Audience"
            color = "#84CC16"
            description = f"3rd Views, #{engagement_rank} Engagement"
            emoji = ""
        elif engagement_rank == 3:
            category = "Moderate Engagement"
            color = "#F59E0B"
            description = f"3rd Engagement, #{views_rank} Views"
            emoji = ""
        
        # Priority 6: Lower performers - differentiate by specific ranks
        elif views_rank == 4:
            category = "Limited Reach"
            color = "#FB923C"
            description = f"4th Views, #{engagement_rank} Engagement"
            emoji = ""
        elif engagement_rank == 4:
            category = "Low Interaction"
            color = "#F97316"
            description = f"4th Engagement, #{views_rank} Views"
            emoji = ""
        
        # Priority 7: Bottom tier - use combination of ranks
        elif views_rank == 5 and engagement_rank == 5:
            category = "Needs Major Boost"
            color = "#DC2626"
            description = f"Lowest: #{engagement_rank} Eng, #{views_rank} Views"
            emoji = ""
        elif views_rank == 5:
            category = "Minimal Reach"
            color = "#EF4444"
            description = f"5th Views, #{engagement_rank} Engagement"
            emoji = ""
        elif engagement_rank == 5:
            category = "Minimal Interaction"
            color = "#DC2626"
            description = f"5th Engagement, #{views_rank} Views"
            emoji = ""
        
        # Fallback: Use unique identifier based on exact rank combination
        else:
            # Create unique category based on rank combination
            category = f"Tier {int(combined_score)}"
            color = "#6B7280"
            description = f"#{engagement_rank} Eng, #{views_rank} Views"
            emoji = ""
        
        insights.append({
            'cluster': cluster_num,
            'category': category,
            'description': description,
            'color': color,
            'emoji': emoji,
            'count': int(cluster_counts[cluster_num]),
            'percentage': float(cluster_counts[cluster_num] / len(df) * 100),
            'avg_likes': float(row['likes']),
            'avg_views': float(row['views']),
            'avg_comments': float(row['comments']),
            'avg_shares': float(row['shares']),
            'avg_engagement': float(row['engagement']),
            'engagement_rank': engagement_rank,
            'views_rank': views_rank,
            'engagement_pct': float(engagement_pct),
            'views_pct': float(views_pct)
        })
    
    # Sort by cluster number
    insights.sort(key=lambda x: x['cluster'])
    
    return insights

def get_content_type_distribution(df):
    """Generate ContentType distribution analysis"""
    if 'ContentType' not in df.columns:
        return None
    
    content_types = []
    
    for content_type in df['ContentType'].unique():
        ct_data = df[df['ContentType'] == content_type]
        
        cluster_dist = ct_data['Cluster'].value_counts()
        dominant_cluster = int(cluster_dist.idxmax())
        dominant_pct = float(cluster_dist.max() / len(ct_data) * 100)
        
        avg_likes = float(ct_data['Likes'].mean())
        avg_views = float(ct_data['Views'].mean())
        avg_comments = float(ct_data['Comments'].mean())
        avg_shares = float(ct_data['Shares'].mean())
        
        age_group_info = "N/A"
        if 'AgeGroup' in df.columns:
            top_age = ct_data['AgeGroup'].mode()
            if len(top_age) > 0:
                age_group_info = str(top_age.iloc[0])
        
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
    
    content_types.sort(key=lambda x: x['count'], reverse=True)
    
    return content_types

def render(df_clustered, result, k_value, features_cols):
    """Render Overview tab with ContentType Distribution"""
    
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
    
    content_type_data = get_content_type_distribution(df_clustered)
    has_content_type = content_type_data is not None
    
    insights_json = json.dumps(insights, default=json_safe)
    distribution_json = json.dumps(distribution_data, default=json_safe)
    bar_chart_json = json.dumps(bar_chart_data, default=json_safe)
    content_type_json = json.dumps(content_type_data, default=json_safe) if has_content_type else json.dumps(None)
    has_content_type_json = json.dumps(has_content_type)
    
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
            .insight-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1rem; margin-bottom: 1.5rem; }}
            .insight-card {{ border-radius: 12px; padding: 1.5rem; border-left: 5px solid; box-shadow: 0 2px 8px rgba(0,0,0,0.08); transition: transform 0.2s; }}
            .insight-card:hover {{ transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.12); }}
            .insight-header {{ display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.8rem; }}
            .insight-cluster {{ font-size: 1.1rem; font-weight: 600; }}
            .insight-emoji {{ font-size: 1.5rem; }}
            .rank-badge {{ display: inline-block; padding: 2px 8px; background: rgba(59, 130, 246, 0.1); color: #3B82F6; border-radius: 4px; font-size: 0.75rem; font-weight: 600; margin-left: 0.5rem; }}
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
            <div class="card"><h3 class="section-header">Distribusi & Performa Cluster</h3></div>
            
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
                <h3 class="section-header">Profil Cluster</h3>
                <p class="section-subtitle">Kategori unik berdasarkan ranking engagement dan views</p>
            </div>
            <div id="insightCards" class="insight-grid"></div>

            <div class="card">
                <h3 class="section-header">Ringkasan Metrik Detail</h3>
                <div style="overflow-x: auto;"><table class="summary-table" id="summaryTable"></table></div>
            </div>

            <div id="contentTypeSection" style="display: none;">
                <div class="card">
                    <h3 class="section-header">Distribusi Content Type</h3>
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
                        <p style="font-weight: 600; color: ${{i.color}}; margin-bottom: 0.3rem;">${{i.category}}</p>
                        <p style="font-size: 0.85rem; color: #64748B; margin-bottom: 0.8rem;">${{i.description}}</p>
                        <div style="display: flex; gap: 0.5rem; flex-wrap: wrap; margin-bottom: 0.8rem;">
                            <span class="rank-badge">Eng: #${{i.engagement_rank}}</span>
                            <span class="rank-badge">Views: #${{i.views_rank}}</span>
                        </div>
                        <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid ${{i.color}}30;">
                            <p><strong>${{i.count.toLocaleString('id-ID')}}</strong> konten</p>
                            <p style="font-size: 0.8rem; color: #64748B;">${{i.percentage.toFixed(1)}}% dari total</p>
                        </div>
                    </div>
                `).join('');
                document.getElementById('insightCards').innerHTML = html;
            }}

            function createSummaryTable() {{
                let html = '<thead><tr><th>Cluster</th><th>Kategori</th><th>Rank</th><th>Jumlah</th><th>%</th><th>Likes</th><th>Views</th><th>Comments</th><th>Shares</th><th>Engagement</th></tr></thead><tbody>';
                insights.forEach(i => {{
                    html += `<tr><td><strong>Cluster ${{i.cluster}}</strong></td><td>${{i.category}}</td><td><span class="rank-badge">E:#${{i.engagement_rank}}</span> <span class="rank-badge">V:#${{i.views_rank}}</span></td><td>${{i.count.toLocaleString('id-ID')}}</td><td>${{i.percentage.toFixed(1)}}%</td><td>${{i.avg_likes.toLocaleString('id-ID', {{maximumFractionDigits: 0}})}}</td><td>${{i.avg_views.toLocaleString('id-ID', {{maximumFractionDigits: 0}})}}</td><td>${{i.avg_comments.toLocaleString('id-ID', {{maximumFractionDigits: 0}})}}</td><td>${{i.avg_shares.toLocaleString('id-ID', {{maximumFractionDigits: 0}})}}</td><td>${{i.avg_engagement.toFixed(4)}}</td></tr>`;
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