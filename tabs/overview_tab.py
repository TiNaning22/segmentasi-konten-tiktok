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
    """Generate insights per cluster dengan dynamic features"""
    insights = []
    
    cluster_means = df.groupby(cluster_col)[features_cols].mean()
    cluster_counts = df[cluster_col].value_counts()
    
    # Prioritaskan features untuk ranking berdasarkan yang tersedia
    engagement_features = []
    view_features = []
    
    # Cari features yang mungkin mewakili engagement
    for feat in features_cols:
        feat_lower = feat.lower()
        if any(keyword in feat_lower for keyword in ['engagement', 'rate', 'interaction', 'like', 'comment', 'share']):
            engagement_features.append(feat)
        elif any(keyword in feat_lower for keyword in ['view', 'reach', 'impression', 'watch']):
            view_features.append(feat)
        elif any(keyword in feat_lower for keyword in ['time', 'duration', 'length']):
            engagement_features.append(feat)
    
    # Default jika tidak ditemukan features spesifik
    if not engagement_features:
        engagement_features = features_cols[:1] if features_cols else []
    if not view_features:
        view_features = features_cols[1:2] if len(features_cols) > 1 else []
    
    # Pilih features utama untuk ranking
    primary_engagement = engagement_features[0] if engagement_features else features_cols[0]
    primary_view = view_features[0] if view_features else (features_cols[1] if len(features_cols) > 1 else features_cols[0])
    
    # Calculate engagement proxy
    engagement_metrics = []
    for cluster_num in sorted(df[cluster_col].unique()):
        avg_metrics = cluster_means.loc[cluster_num]
        
        # Pastikan avg_metrics adalah Series, bukan DataFrame
        if isinstance(avg_metrics, pd.DataFrame):
            avg_metrics = avg_metrics.iloc[0]
        
        # Get scalar values untuk perbandingan
        engagement_rate = 0
        if primary_engagement in avg_metrics.index:
            engagement_val = avg_metrics[primary_engagement]
            # Pastikan ini scalar, bukan array
            if isinstance(engagement_val, (pd.Series, np.ndarray)):
                engagement_rate = float(engagement_val.iloc[0]) if hasattr(engagement_val, 'iloc') else float(engagement_val[0])
            else:
                engagement_rate = float(engagement_val)
        
        # Get view metric
        views = 0
        if primary_view in avg_metrics.index:
            view_val = avg_metrics[primary_view]
            if isinstance(view_val, (pd.Series, np.ndarray)):
                views = float(view_val.iloc[0]) if hasattr(view_val, 'iloc') else float(view_val[0])
            else:
                views = float(view_val)
        
        # Get other metrics jika ada
        likes = 0
        if 'Likes' in avg_metrics.index:
            like_val = avg_metrics['Likes']
            if isinstance(like_val, (pd.Series, np.ndarray)):
                likes = float(like_val.iloc[0]) if hasattr(like_val, 'iloc') else float(like_val[0])
            else:
                likes = float(like_val)
        
        comments = 0
        if 'Comments' in avg_metrics.index:
            comment_val = avg_metrics['Comments']
            if isinstance(comment_val, (pd.Series, np.ndarray)):
                comments = float(comment_val.iloc[0]) if hasattr(comment_val, 'iloc') else float(comment_val[0])
            else:
                comments = float(comment_val)
        
        shares = 0
        if 'Shares' in avg_metrics.index:
            share_val = avg_metrics['Shares']
            if isinstance(share_val, (pd.Series, np.ndarray)):
                shares = float(share_val.iloc[0]) if hasattr(share_val, 'iloc') else float(share_val[0])
            else:
                shares = float(share_val)
        
        engagement_metrics.append({
            'cluster': cluster_num,
            'engagement': engagement_rate,
            'views': views,
            'likes': likes,
            'comments': comments,
            'shares': shares
        })
    
    # Convert to DataFrame untuk ranking
    df_metrics = pd.DataFrame(engagement_metrics)
    
    # Handle ranking dengan aman
    if len(df_metrics) > 1 and 'engagement' in df_metrics.columns and df_metrics['engagement'].notna().all():
        df_metrics['engagement_rank'] = df_metrics['engagement'].rank(ascending=False, method='min')
    else:
        df_metrics['engagement_rank'] = 1
    
    if len(df_metrics) > 1 and 'views' in df_metrics.columns and df_metrics['views'].notna().all():
        df_metrics['views_rank'] = df_metrics['views'].rank(ascending=False, method='min')
    else:
        df_metrics['views_rank'] = 1
    
    # Calculate percentiles dengan handling NaN
    if len(df_metrics) > 1:
        if 'engagement' in df_metrics.columns:
            df_metrics['engagement_pct'] = df_metrics['engagement'].rank(pct=True, na_option='keep')
            df_metrics['engagement_pct'] = df_metrics['engagement_pct'].fillna(0.5)
        if 'views' in df_metrics.columns:
            df_metrics['views_pct'] = df_metrics['views'].rank(pct=True, na_option='keep')
            df_metrics['views_pct'] = df_metrics['views_pct'].fillna(0.5)
    else:
        df_metrics['engagement_pct'] = 1.0
        df_metrics['views_pct'] = 1.0
    
    # Assign categories - BAHASA INDONESIA
    for idx, row in df_metrics.iterrows():
        cluster_num = row['cluster']
        engagement_rank = int(row['engagement_rank'])
        views_rank = int(row['views_rank'])
        engagement_pct = row.get('engagement_pct', 0.5)
        views_pct = row.get('views_pct', 0.5)
        
        # Gunakan combined score
        combined_score = (engagement_rank + views_rank) / 2
        
        # Kategorisasi dalam Bahasa Indonesia
        if engagement_rank == 1 and views_rank == 1:
            category = "Performa Terbaik"
            color = "#10B981"
            description = "Terbaik dalam Engagement & Jangkauan"
            emoji = ""
        elif engagement_rank == 1:
            category = "Engagement Tinggi"
            color = "#F59E0B"
            description = "Engagement Tertinggi, Jangkauan Menengah"
            emoji = ""
        elif views_rank == 1:
            category = "Jangkauan Luas"
            color = "#3B82F6"
            description = "Jangkauan Tertinggi, Engagement Menengah"
            emoji = ""
        elif combined_score <= 2.5:
            category = "Seimbang Kuat"
            color = "#8B5CF6"
            description = "Engagement & Jangkauan Baik"
            emoji = ""
        elif combined_score <= 4.0:
            category = "Rata-rata"
            color = "#64748B"
            description = "Performa Menengah"
            emoji = ""
        else:
            category = "Perlu Peningkatan"
            color = "#EF4444"
            description = "Engagement & Jangkauan Rendah"
            emoji = ""
        
        # Get cluster data
        cluster_data = df[df[cluster_col] == cluster_num]
        
        insights.append({
            'cluster': cluster_num,
            'category': category,
            'description': description,
            'color': color,
            'emoji': emoji,
            'count': int(cluster_counts.get(cluster_num, 0)),
            'percentage': float(cluster_counts.get(cluster_num, 0) / len(df) * 100) if len(df) > 0 else 0,
            'avg_engagement': float(row['engagement']),
            'avg_views': float(row['views']),
            'avg_likes': float(row['likes']),
            'avg_comments': float(row['comments']),
            'avg_shares': float(row['shares']),
            'engagement_rank': engagement_rank,
            'views_rank': views_rank,
            'engagement_pct': float(engagement_pct),
            'views_pct': float(views_pct),
            'primary_features': {
                'engagement': primary_engagement,
                'views': primary_view
            }
        })
    
    # Sort by cluster number
    insights.sort(key=lambda x: x['cluster'])
    
    return insights

def get_content_type_distribution(df):
    """Generate ContentType distribution analysis dengan dynamic features"""
    if 'ContentType' not in df.columns:
        # Coba cari kolom kategorikal lainnya
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        # Pilih kolom kategorikal pertama yang bukan 'Cluster'
        for col in categorical_cols:
            if col != 'Cluster' and df[col].nunique() <= 10:  # Batasi unique values
                st.info(f"Menggunakan '{col}' untuk analisis tipe konten")
                content_types = []
                
                for cat_value in df[col].dropna().unique():
                    ct_data = df[df[col] == cat_value]
                    
                    if len(ct_data) < 3:  # Skip jika terlalu sedikit data
                        continue
                    
                    cluster_dist = ct_data['Cluster'].value_counts()
                    dominant_cluster = int(cluster_dist.idxmax()) if len(cluster_dist) > 0 else 0
                    dominant_pct = float(cluster_dist.max() / len(ct_data) * 100) if len(cluster_dist) > 0 else 0
                    
                    # Calculate average metrics yang tersedia
                    avg_metrics = {}
                    numeric_cols = ct_data.select_dtypes(include=[np.number]).columns
                    for metric in ['Likes', 'Views', 'Comments', 'Shares', 'Engagement_Rate']:
                        if metric in numeric_cols:
                            avg_metrics[metric] = float(ct_data[metric].mean())
                    
                    # Determine performance - BAHASA INDONESIA
                    performance = "Menengah"
                    perf_color = "#F59E0B"
                    
                    content_types.append({
                        'content_type': str(cat_value),
                        'count': int(len(ct_data)),
                        'percentage': float(len(ct_data) / len(df) * 100),
                        **avg_metrics,
                        'dominant_cluster': dominant_cluster,
                        'dominant_cluster_pct': dominant_pct,
                        'performance': performance,
                        'performance_color': perf_color
                    })
                
                content_types.sort(key=lambda x: x['count'], reverse=True)
                return content_types
        
        return None
    
    # Original logic jika ContentType ada
    content_types = []
    
    for content_type in df['ContentType'].dropna().unique():
        ct_data = df[df['ContentType'] == content_type]
        
        cluster_dist = ct_data['Cluster'].value_counts()
        dominant_cluster = int(cluster_dist.idxmax())
        dominant_pct = float(cluster_dist.max() / len(ct_data) * 100)
        
        # Gunakan hanya metrics yang ada
        avg_metrics = {}
        available_metrics = ['Likes', 'Views', 'Comments', 'Shares', 'Engagement_Rate']
        for metric in available_metrics:
            if metric in ct_data.columns and pd.api.types.is_numeric_dtype(ct_data[metric]):
                avg_metrics[metric] = float(ct_data[metric].mean())
        
        age_group_info = "N/A"
        if 'AgeGroup' in df.columns:
            top_age = ct_data['AgeGroup'].mode()
            if len(top_age) > 0:
                age_group_info = str(top_age.iloc[0])
        
        # Calculate engagement rate dari metrics yang tersedia
        engagement_rate = 0
        if 'Engagement_Rate' in avg_metrics:
            engagement_rate = avg_metrics['Engagement_Rate']
        elif all(m in avg_metrics for m in ['Likes', 'Comments', 'Shares', 'Views']):
            engagement_rate = (avg_metrics['Likes'] + avg_metrics['Comments'] + avg_metrics['Shares']) / max(avg_metrics['Views'], 1)
        
        # Performance categories - BAHASA INDONESIA
        if engagement_rate >= 0.05 and 'Views' in avg_metrics and avg_metrics['Views'] >= df['Views'].median():
            performance = "Tinggi"
            perf_color = "#10B981"
        elif engagement_rate >= 0.02 or ('Views' in avg_metrics and avg_metrics['Views'] >= df['Views'].median()):
            performance = "Menengah"
            perf_color = "#F59E0B"
        else:
            performance = "Rendah"
            perf_color = "#EF4444"
        
        content_types.append({
            'content_type': str(content_type),
            'count': int(len(ct_data)),
            'percentage': float(len(ct_data) / len(df) * 100),
            **avg_metrics,
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
    """Render Overview tab dengan dynamic features"""
    
    insights = get_cluster_insights(df_clustered, 'Cluster', features_cols)
    
    cluster_counts = df_clustered['Cluster'].value_counts().sort_index()
    distribution_data = {
        'labels': [f"Cluster {i}" for i in cluster_counts.index],
        'values': cluster_counts.values.tolist()
    }
    
    # Pilih max 4 features untuk bar chart
    main_features = features_cols[:4] if len(features_cols) >= 4 else features_cols
    
    cluster_means = df_clustered.groupby('Cluster')[main_features].mean()
    
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
            .insight-card {{ border-radius: 12px; padding: 1.5rem; box-shadow: 0 2px 8px rgba(0,0,0,0.08); transition: transform 0.2s; }}
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
            .feature-info {{ font-size: 0.8rem; color: #64748B; margin-top: 0.5rem; font-style: italic; }}
            @media (max-width: 768px) {{ .two-column, .insight-grid {{ grid-template-columns: 1fr; }} }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="card">
                <h3 class="section-header">Distribusi & Performa Cluster</h3>
                <p class="section-subtitle">
                    Features yang digunakan: {', '.join(features_cols[:5])}{'...' if len(features_cols) > 5 else ''}
                </p>
            </div>
            
            <div class="two-column">
                <div class="card">
                    <h4 style="font-size: 1rem; font-weight: 600; margin-bottom: 1rem;">Distribusi Konten per Cluster</h4>
                    <div id="pieChart" class="chart-container"></div>
                </div>
                <div class="card">
                    <h4 style="font-size: 1rem; font-weight: 600; margin-bottom: 1rem;">Rata-rata Features per Cluster</h4>
                    <div class="feature-info">
                        Menampilkan: {', '.join(main_features)}
                    </div>
                    <div id="barChart" class="chart-container"></div>
                </div>
            </div>

            <div class="card">
                <h3 class="section-header">Profil Cluster</h3>
                <p class="section-subtitle">
                    Kategori berdasarkan ranking {insights[0]['primary_features']['engagement'] if insights else 'engagement'} dan {insights[0]['primary_features']['views'] if insights else 'views'}
                </p>
            </div>
            <div id="insightCards" class="insight-grid"></div>

            <div class="card">
                <h3 class="section-header">Ringkasan Metrik Detail</h3>
                <div style="overflow-x: auto;"><table class="summary-table" id="summaryTable"></table></div>
            </div>

            <div id="contentTypeSection" style="display: none;">
                <div class="card">
                    <h3 class="section-header">Distribusi Tipe Konten</h3>
                    <p class="section-subtitle">Performa berdasarkan jenis konten dengan cluster dominan</p>
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
                    marker: {{ color: ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6', '#EC4899'][idx] }}
                }}));
                Plotly.newPlot('barChart', traces, {{ 
                    barmode: 'group', 
                    height: 420, 
                    yaxis: {{ title: 'Nilai Rata-rata' }}, 
                    margin: {{ t: 60, b: 40, l: 60, r: 20 }}, 
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    legend: {{ orientation: 'h', y: 1.1 }}
                }}, {{ responsive: true, displayModeBar: false }});
            }}

            function createInsightCards() {{
                const html = insights.map(i => `
                    <div class="insight-card" style="background: linear-gradient(135deg, ${{i.color}}08 0%, ${{i.color}}15 100%); border-left: 4px solid ${{i.color}};">
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
                            <p style="font-size: 0.75rem; color: #94A3B8; margin-top: 0.3rem;">
                                ${{i.primary_features.engagement}}: ${{i.avg_engagement.toFixed(4)}}
                            </p>
                        </div>
                    </div>
                `).join('');
                document.getElementById('insightCards').innerHTML = html;
            }}

            function createSummaryTable() {{
                let html = '<thead><tr><th>Cluster</th><th>Kategori</th><th>Peringkat</th><th>Jumlah</th><th>%</th>';
                
                // Tambah kolom untuk features yang ada
                const sampleInsight = insights[0];
                if (sampleInsight.avg_engagement !== undefined) html += '<th>Engagement</th>';
                if (sampleInsight.avg_views !== undefined) html += '<th>Views</th>';
                if (sampleInsight.avg_likes !== undefined && sampleInsight.avg_likes > 0) html += '<th>Likes</th>';
                if (sampleInsight.avg_comments !== undefined && sampleInsight.avg_comments > 0) html += '<th>Comments</th>';
                if (sampleInsight.avg_shares !== undefined && sampleInsight.avg_shares > 0) html += '<th>Shares</th>';
                
                html += '</tr></thead><tbody>';
                
                insights.forEach(i => {{
                    html += `<tr>
                        <td><strong>Cluster ${{i.cluster}}</strong></td>
                        <td>${{i.category}}</td>
                        <td><span class="rank-badge">E:#${{i.engagement_rank}}</span> <span class="rank-badge">V:#${{i.views_rank}}</span></td>
                        <td>${{i.count.toLocaleString('id-ID')}}</td>
                        <td>${{i.percentage.toFixed(1)}}%</td>`;
                    
                    if (i.avg_engagement !== undefined) html += `<td>${{i.avg_engagement.toFixed(4)}}</td>`;
                    if (i.avg_views !== undefined) html += `<td>${{i.avg_views.toLocaleString('id-ID', {{maximumFractionDigits: 0}})}}</td>`;
                    if (i.avg_likes !== undefined && i.avg_likes > 0) html += `<td>${{i.avg_likes.toLocaleString('id-ID', {{maximumFractionDigits: 0}})}}</td>`;
                    if (i.avg_comments !== undefined && i.avg_comments > 0) html += `<td>${{i.avg_comments.toLocaleString('id-ID', {{maximumFractionDigits: 0}})}}</td>`;
                    if (i.avg_shares !== undefined && i.avg_shares > 0) html += `<td>${{i.avg_shares.toLocaleString('id-ID', {{maximumFractionDigits: 0}})}}</td>`;
                    
                    html += '</tr>';
                }});
                
                document.getElementById('summaryTable').innerHTML = html + '</tbody>';
            }}

            function createContentTypeTable() {{
                if (!hasContentType || !contentTypeData) return;
                document.getElementById('contentTypeSection').style.display = 'block';
                
                let html = '<thead><tr><th>Content Type</th><th>Jumlah</th><th>%</th>';
                
                // Add columns untuk metrics yang tersedia
                const sampleCt = contentTypeData[0];
                if (sampleCt.Likes !== undefined) html += '<th>Likes</th>';
                if (sampleCt.Views !== undefined) html += '<th>Views</th>';
                if (sampleCt.Comments !== undefined) html += '<th>Comments</th>';
                if (sampleCt.Shares !== undefined) html += '<th>Shares</th>';
                if (sampleCt.Engagement_Rate !== undefined) html += '<th>Eng Rate</th>';
                
                html += '<th>Cluster Dominan</th><th>Performance</th></tr></thead><tbody>';
                
                contentTypeData.forEach(ct => {{
                    html += `<tr style="border-left: 3px solid ${{ct.performance_color}};">
                        <td><strong>${{ct.content_type}}</strong><div class="mini-bar" style="width: ${{ct.percentage}}%; background: linear-gradient(90deg, ${{ct.performance_color}} 0%, ${{ct.performance_color}}80 100%);"></div></td>
                        <td style="text-align: right; font-weight: 600;">${{ct.count.toLocaleString('id-ID')}}</td>
                        <td style="text-align: right;">${{ct.percentage.toFixed(1)}}%</td>`;
                    
                    if (ct.Likes !== undefined) html += `<td style="text-align: right;">${{ct.Likes.toLocaleString('id-ID', {{maximumFractionDigits: 0}})}}</td>`;
                    if (ct.Views !== undefined) html += `<td style="text-align: right;">${{ct.Views.toLocaleString('id-ID', {{maximumFractionDigits: 0}})}}</td>`;
                    if (ct.Comments !== undefined) html += `<td style="text-align: right;">${{ct.Comments.toLocaleString('id-ID', {{maximumFractionDigits: 0}})}}</td>`;
                    if (ct.Shares !== undefined) html += `<td style="text-align: right;">${{ct.Shares.toLocaleString('id-ID', {{maximumFractionDigits: 0}})}}</td>`;
                    if (ct.Engagement_Rate !== undefined) html += `<td style="text-align: right;">${{ct.Engagement_Rate.toFixed(4)}}</td>`;
                    
                    html += `
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