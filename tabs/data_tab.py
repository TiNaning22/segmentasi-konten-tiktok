import streamlit as st
import pandas as pd

def render(df_clustered, result, k_value, features_cols):
    """Render Data tab"""
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown("#### 📊 Data dengan Label Cluster")
        
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            selected_clusters = st.multiselect(
                "Filter Cluster:",
                options=sorted(df_clustered['Cluster'].unique()),
                default=sorted(df_clustered['Cluster'].unique()),
                format_func=lambda x: f"Cluster {x}"
            )
        
        with col_b:
            sort_by = st.selectbox(
                "Urutkan berdasarkan:",
                features_cols
            )
        
        with col_c:
            display_limit = st.number_input(
                "Tampilkan rows:",
                min_value=10,
                max_value=1000,
                value=100,
                step=10
            )
        
        df_filtered = df_clustered[df_clustered['Cluster'].isin(selected_clusters)].sort_values(
            by=sort_by, 
            ascending=False
        ).head(display_limit)
        
        st.dataframe(
            df_filtered,
            use_container_width=True,
            height=400
        )
        
        csv = df_clustered.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=f"📥 Download Data (K={k_value})",
            data=csv,
            file_name=f"tiktok_clustered_k{k_value}.csv",
            mime="text/csv",
            type="primary"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown("#### ⚙️ Settings")
        
        color_scheme = st.selectbox(
            "Color Scheme:",
            ["Set3", "Pastel", "Plotly", "D3", "Viridis"],
            index=0
        )
        
        show_detailed = st.checkbox("Show Detailed Stats", value=False)
        
        st.markdown("---")
        st.markdown("**Cluster Colors:**")
        import plotly.express as px
        colors = px.colors.qualitative.Set3[:k_value]
        for i, color in enumerate(colors):
            st.markdown(f'<div style="background-color:{color}; padding: 5px; border-radius: 3px; margin: 2px 0;">Cluster {i}</div>', 
                      unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)