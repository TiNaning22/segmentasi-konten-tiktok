import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def render(df_clustered, result, k_value, features_cols):
    """Render Visualization tab"""
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown("#### Visualisasi 2D Cluster (PCA)")
        
        # Perbaikan: Cek apakah sample_indices tidak None
        if result['use_sample'] and result['sample_indices'] is not None:
            sample_idx = result['sample_indices']
            pca_df = pd.DataFrame({
                'PC1': result['pca_result'][sample_idx, 0],
                'PC2': result['pca_result'][sample_idx, 1],
                'Cluster': [f"Cluster {c}" for c in result['clusters'][sample_idx]],
                'Likes': df_clustered.iloc[sample_idx]['Likes'].values,
                'Views': df_clustered.iloc[sample_idx]['Views'].values,
            })
        else:
            # Gunakan semua data tanpa indexing
            pca_df = pd.DataFrame({
                'PC1': result['pca_result'][:, 0],
                'PC2': result['pca_result'][:, 1],
                'Cluster': [f"Cluster {c}" for c in result['clusters']],
                'Likes': df_clustered['Likes'].values,
                'Views': df_clustered['Views'].values,
            })
        
        fig_scatter = px.scatter(
            pca_df,
            x='PC1',
            y='PC2',
            color='Cluster',
            hover_data=['Likes', 'Views'],
            color_discrete_sequence=px.colors.qualitative.Set3,
            title=f"Scatter Plot Cluster (K={k_value})"
        )
        
        fig_scatter.update_layout(height=500)
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        st.info(f"""
        **PCA Explained Variance:**  
        PC1 = {result['pca_explained'][0]*100:.1f}%, PC2 = {result['pca_explained'][1]*100:.1f}%  
        **Total:** {sum(result['pca_explained'])*100:.1f}% variance dijelaskan
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown("#### Quick Stats")
        
        st.markdown("**Varian per Cluster:**")
        for cluster_num in sorted(df_clustered['Cluster'].unique()):
            cluster_data = df_clustered[df_clustered['Cluster'] == cluster_num]
            st.markdown(f"""
            **Cluster {cluster_num}:**
            - Avg Likes: {cluster_data['Likes'].mean():,.0f}
            - Avg Views: {cluster_data['Views'].mean():,.0f}
            - Engagement: {cluster_data['Engagement_Rate'].mean():.4f}
            """)
        st.markdown('</div>', unsafe_allow_html=True)