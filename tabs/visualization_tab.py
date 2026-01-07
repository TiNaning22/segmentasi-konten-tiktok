import streamlit as st
import plotly.express as px
import pandas as pd


def render(df_clustered, result, k_value, features_cols):
    """Render Visualization tab (HTML Hybrid)"""

    # =======================
    # GLOBAL CSS
    # =======================
    st.markdown("""
    <style>
    .custom-card {
        background-color: #FFFFFF;
        border-radius: 14px;
        padding: 18px;
        margin-bottom: 16px;
    }
    .card-title {
        color: #1E293B;
        font-size: 18px;
        font-weight: 600;
        margin-bottom: 12px;
    }
    .card-text {
        color: #1E293B;
        font-size: 14px;
        line-height: 1.6;
    }
    .cluster-box {
        background: rgba(255,255,255,0.04);
        padding: 10px 12px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])

    # =======================
    # LEFT COLUMN – PCA SCATTER
    # =======================
    with col1:
        st.markdown("""
        <div class="custom-card">
            <div class="card-title">Visualisasi 2D Cluster (PCA)</div>
        """, unsafe_allow_html=True)

        # ----- DATA PREPARATION (TIDAK DIUBAH) -----
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
            pca_df = pd.DataFrame({
                'PC1': result['pca_result'][:, 0],
                'PC2': result['pca_result'][:, 1],
                'Cluster': [f"Cluster {c}" for c in result['clusters']],
                'Likes': df_clustered['Likes'].values,
                'Views': df_clustered['Views'].values,
            })

        # ----- SCATTER PLOT (TETAP SAMA) -----
        fig_scatter = px.scatter(
            pca_df,
            x='PC1',
            y='PC2',
            color='Cluster',
            hover_data=['Likes', 'Views'],
            color_discrete_sequence=px.colors.qualitative.Set3,
            title=f"Scatter Plot Cluster (K={k_value})"
        )

        fig_scatter.update_layout(
            height=500,
            plot_bgcolor="#0f172a",
            paper_bgcolor="#0f172a",
            font=dict(color="white")
        )

        st.plotly_chart(fig_scatter, use_container_width=True)

        st.markdown(f"""
        <div class="card-text">
            <b>PCA Explained Variance</b><br>
            PC1 = {result['pca_explained'][0]*100:.1f}% <br>
            PC2 = {result['pca_explained'][1]*100:.1f}% <br>
            <b>Total:</b> {sum(result['pca_explained'])*100:.1f}%
        </div>
        """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # =======================
    # RIGHT COLUMN – QUICK STATS
    # =======================
    with col2:
        st.markdown("""
        <div class="custom-card">
            <div class="card-title">Quick Stats</div>
            <div class="card-text"><b>Varian per Cluster</b></div>
        """, unsafe_allow_html=True)

        for cluster_num in sorted(df_clustered['Cluster'].unique()):
            cluster_data = df_clustered[df_clustered['Cluster'] == cluster_num]

            st.markdown(f"""
            <div class="cluster-box card-text">
                <b>Cluster {cluster_num}</b><br>
                Avg Likes : {cluster_data['Likes'].mean():,.0f}<br>
                Avg Views : {cluster_data['Views'].mean():,.0f}<br>
                Engagement : {cluster_data['Engagement_Rate'].mean():.4f}
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)
