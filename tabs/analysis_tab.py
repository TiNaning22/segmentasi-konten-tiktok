import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from scipy.spatial.distance import cdist

def render(df_clustered, result, k_value, features_cols):
    """Render Analysis tab"""
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown("#### 📈 Analisis Mendalam")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Cluster Centers (Centroid)**")
        
        centers = result['scaler'].inverse_transform(result['kmeans'].cluster_centers_)
        centers_df = pd.DataFrame(
            centers,
            columns=features_cols,
            index=[f"Cluster {i}" for i in range(k_value)]
        )
        
        st.dataframe(
            centers_df.style.background_gradient(cmap='Blues', axis=0).format("{:.2f}"),
            use_container_width=True
        )
    
    with col2:
        st.markdown("**Metrik Trend per Cluster**")
        
        selected_feature = st.selectbox("Pilih metrik:", features_cols, key='trend_feature')
        
        fig_trend = go.Figure()
        
        for cluster_num in sorted(df_clustered['Cluster'].unique()):
            cluster_data = df_clustered[df_clustered['Cluster'] == cluster_num]
            
            fig_trend.add_trace(go.Box(
                y=cluster_data[selected_feature],
                name=f"Cluster {cluster_num}",
                boxpoints='outliers'
            ))
        
        fig_trend.update_layout(
            height=350,
            xaxis_title="Cluster",
            yaxis_title=selected_feature,
            showlegend=False
        )
        
        st.plotly_chart(fig_trend, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)