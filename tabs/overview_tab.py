import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

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
        elif views_pct >= 0.75:
            category = "Passive Viral"
            color = "#3B82F6"
            description = "Views Tinggi, Interaksi Rendah"
        elif engagement_pct >= 0.75:
            category = "Conversation Starter"
            color = "#F59E0B"
            description = "Engagement Tinggi, Reach Sedang"
        elif engagement_pct >= 0.50 and views_pct >= 0.50:
            category = "Moderate Performer"
            color = "#6B7280"
            description = "Performa Rata-rata"
        else:
            category = "Underperforming"
            color = "#EF4444"
            description = "Perlu Optimasi"
        
        insights.append({
            'cluster': cluster_num,
            'category': category,
            'description': description,
            'color': color,
            'count': cluster_counts[cluster_num],
            'percentage': cluster_counts[cluster_num] / len(df) * 100,
            'avg_metrics': avg_metrics,
            'engagement_rank': engagement_pct,
            'views_rank': views_pct
        })
    
    return insights

def render(df_clustered, result, k_value, features_cols):
    """Render Overview tab"""
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown("#### Distribusi Konten per Cluster")
        
        cluster_counts = df_clustered['Cluster'].value_counts().sort_index()
        
        fig_pie = px.pie(
            values=cluster_counts.values,
            names=[f"Cluster {i}" for i in cluster_counts.index],
            color_discrete_sequence=px.colors.qualitative.Set3,
            hole=0.4
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        fig_pie.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig_pie, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown("#### Rata-rata Metrik per Cluster")
        
        cluster_means = df_clustered.groupby('Cluster')[features_cols].mean()
        
        fig_bar = go.Figure()
        
        for feature in features_cols[:4]:
            fig_bar.add_trace(go.Bar(
                name=feature,
                x=[f"Cluster {i}" for i in cluster_means.index],
                y=cluster_means[feature],
                text=cluster_means[feature].round(2),
                textposition='auto'
            ))
        
        fig_bar.update_layout(
            barmode='group',
            height=350,
            xaxis_title="Cluster",
            yaxis_title="Nilai Rata-rata",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig_bar, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Cluster Insights
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown("#### 🎯 Profil Cluster (Interpretasi Otomatis)")
    
    insights = get_cluster_insights(df_clustered, 'Cluster', features_cols)
    
    cols = st.columns(min(k_value, 4))
    
    for idx, insight in enumerate(insights):
        with cols[idx % 4]:
            st.markdown(f"""
            <div style='
                padding: 15px; 
                border-radius: 10px; 
                background-color: {insight['color']}15; 
                border-left: 4px solid {insight['color']};
                margin-bottom: 10px;
            '>
                <h4 style='margin: 0; color: #1E293B;'>Cluster {insight['cluster']}</h4>
                <p style='font-size: 16px; margin: 5px 0; color: {insight['color']}; font-weight: 600;'>
                    {insight['category']}
                </p>
                <p style='font-size: 14px; color: #64748B; margin: 5px 0;'>{insight['description']}</p>
                <p style='margin: 5px 0; color: #475569;'>
                    <strong>{insight['count']}</strong> konten ({insight['percentage']:.1f}%)
                </p>
            </div>
            """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)