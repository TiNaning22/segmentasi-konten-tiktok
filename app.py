import streamlit as st
import pandas as pd
import numpy as np
import warnings
import logging
from typing import Dict, Any, Optional, Tuple, List

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tiktok_dashboard.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

# Import custom modules
from utils.css_loader import load_css
from utils.data_loader import load_data
from utils.clustering import perform_clustering
from utils.validators import validate_data_for_clustering, suggest_optimal_clusters
from utils.diagnostics import display_clustering_diagnostics

# Import tab modules
from tabs import (
    overview_tab, 
    visualization_tab, 
    data_tab, 
    categorical_tab, 
    analysis_tab
)

# ==================== KONFIGURASI ====================
st.set_page_config(
    page_title="TikTok Content Segmenter",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==================== DASHBOARD UTAMA ====================
def main_dashboard():
    # Initialize session state
    if 'clustering_complete' not in st.session_state:
        st.session_state.clustering_complete = False
    if 'df_clustered' not in st.session_state:
        st.session_state.df_clustered = None
    if 'result' not in st.session_state:
        st.session_state.result = None
        
    # Load CSS
    load_css()
    
    # ==================== HEADER ====================
    st.markdown("""
    <div class='header-card'>
        <h1 style='color: #1E293B; margin: 0 0 0.5rem 0;'>
            TikTok Intelligence Dashboard
        </h1>
        <p style='color: #64748B; font-size: 1rem; margin: 0;'>
            Segmentasi Konten Otomatis Menggunakan K-Means Clustering
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # ==================== KONTROL & INFO ====================
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class='custom-card'>
            <h3 style='color: #1E293B; margin: 0 0 1rem 0;'>
                Kontrol Clustering
            </h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Get suggestions for optimal K
        df_loaded = False
        try:
            df = load_data()
            df_loaded = True
            features_cols = ['Likes', 'Shares', 'Comments', 'Views', 'TimeSpentOnContent', 'Engagement_Rate']
            suggestions = suggest_optimal_clusters(df, features_cols)
        except:
            df_loaded = False
            suggestions = {'recommended': 4, 'recommended_range': '2-5'}
        
        # K value slider with suggestions - MAX VALUE CHANGED TO 5
        default_k = min(suggestions.get('recommended', 4), 5)  # Ensure default doesn't exceed max
        
        k_value = st.slider(
            "Jumlah Cluster (K):",
            min_value=2,
            max_value=5,  # Changed from 10 to 5
            value=default_k,
            help=f"Disarankan: {suggestions.get('recommended_range', '2-5')}. K={default_k} berdasarkan data"
        )
        
        # Show suggestions
        if suggestions.get('warnings'):
            with st.expander("Saran & Peringatan"):
                for warning in suggestions['warnings']:
                    st.warning(warning)
                
                if suggestions.get('recommended_range'):
                    st.info(f"Range yang disarankan: {suggestions['recommended_range']}")
    
    with col2:
        if not df_loaded:
            st.error("Tidak dapat memuat data. Cek file dataset.")
            st.stop()
        
        df = load_data()
        features_cols = ['Likes', 'Shares', 'Comments', 'Views', 'TimeSpentOnContent', 'Engagement_Rate']
        
        missing_values = df[features_cols].isnull().sum().sum()
        
        st.markdown(f"""
        <div class='custom-card'>
            <h3 style='color: #1E293B; margin: 0 0 1rem 0;'>
                Dataset Information
            </h3>
            <div class='info-grid'>
                <div class='info-item'>
                    <p class='info-label'>Total Konten</p>
                    <p class='info-value'>{len(df):,}</p>
                </div>
                <div class='info-item'>
                    <p class='info-label'>Features</p>
                    <p class='info-value'>{len(features_cols)}</p>
                </div>
                <div class='info-item'>
                    <p class='info-label'>Missing Values</p>
                    <p class='info-value {'success' if missing_values == 0 else 'warning' if missing_values < 10 else 'error'}">
                        {missing_values}
                    </p>
                </div>
                <div class='info-item'>
                    <p class='info-label'>Engagement Rate</p>
                    <p class='info-value success'>Aktif</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick data validation
        is_valid, validation_msg, validation_warnings = validate_data_for_clustering(df, features_cols)
        
        if not is_valid:
            st.error(validation_msg)
        else:
            if validation_warnings:
                with st.expander("Validation Warnings", expanded=False):
                    for warning in validation_warnings[:3]:
                        st.warning(warning)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ==================== CLUSTERING DENGAN ERROR HANDLING ====================
    with st.spinner("Melakukan clustering..."):
        progress_bar = st.progress(0)
        
        try:
            # Update progress
            progress_bar.progress(20)
            
            # Validasi data sebelum clustering
            is_valid, validation_msg, validation_warnings = validate_data_for_clustering(df, features_cols)
            
            if not is_valid:
                progress_bar.progress(100)
                st.error(f"{validation_msg}")
                
                with st.expander("Tips perbaikan data"):
                    st.markdown("""
                    1. **Cek missing values**: Pastikan tidak ada kolom dengan >50% missing
                    2. **Cek tipe data**: Semua feature harus numerik
                    3. **Cek variance**: Pastikan features memiliki variasi data
                    4. **Cek outliers**: Outlier ekstrem dapat mempengaruhi clustering
                    5. **Minimum data**: Minimal 10 data points untuk clustering
                    """)
                
                # Suggest optimal clusters
                suggestions = suggest_optimal_clusters(df, features_cols)
                recommended_k = min(suggestions.get('recommended', 4), 5)  # Cap at 5
                st.info(f"Saran: Untuk {len(df):,} data, coba K={recommended_k}")
                
                st.stop()
            
            progress_bar.progress(40)
            
            # Tampilkan info validasi
            if validation_warnings:
                st.warning(f"{len(validation_warnings)} warnings ditemukan. Clustering tetap dilanjutkan.")
            
            progress_bar.progress(60)
            
            # Lakukan clustering
            result = perform_clustering(df, k_value, features_cols)
            
            progress_bar.progress(80)
            
            # Cek jika clustering menghasilkan error
            if result and not result.get('success', True):
                st.error(f"Error dalam clustering: {result.get('error', 'Unknown error')}")
                
                # Fallback: coba dengan K yang lebih kecil
                if k_value > 2:
                    st.warning(f"Mencoba clustering dengan K={k_value-1}...")
                    result = perform_clustering(df, k_value-1, features_cols)
                    
                    if not result.get('success', True):
                        st.error("Clustering tetap gagal. Silakan cek data Anda.")
                        st.stop()
                    else:
                        st.success(f"Clustering berhasil dengan K={k_value-1}")
                        k_value = k_value - 1  # Update K value
                else:
                    st.error("Tidak dapat melakukan clustering. Cek data dan coba lagi.")
                    st.stop()
            
            progress_bar.progress(100)
            
        except Exception as e:
            progress_bar.progress(100)
            logger.error(f"Exception tidak terduga: {str(e)}", exc_info=True)
            st.error(f"Exception tidak terduga: {str(e)}")
            
            # Fallback: buat cluster dummy untuk menjaga aplikasi tetap berjalan
            st.warning("Membuat cluster dummy untuk melanjutkan...")
            result = {
                'clusters': np.random.randint(0, k_value, size=len(df)),
                'metrics': {
                    'silhouette': 0.1,
                    'davies_bouldin': 5.0,
                    'inertia': 0,
                    'cluster_sizes': np.ones(k_value) * (len(df) // k_value),
                    'cluster_balance': 0.5
                },
                'pca_result': np.random.randn(len(df), 2),
                'pca_explained': [0.5, 0.3],
                'success': False,
                'fallback': True,
                'error': str(e)
            }
    
    # ==================== PREPARE CLUSTERED DATA ====================
    df_clustered = df.copy()
    df_clustered['Cluster'] = result['clusters']
    
    # ==================== DIAGNOSTICS ====================
    # display_clustering_diagnostics(df, result, features_cols)
    
    # ==================== METRICS DENGAN ERROR INDICATOR ====================
    st.markdown("""
    <div class='custom-card'>
        <h3 style='color: #1E293B; margin: 0 0 1.5rem 0;'>
            Model Performance Metrics
        </h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Tampilkan indicator jika ada error atau fallback
    if result.get('fallback', False):
        st.warning("Menggunakan hasil fallback clustering. Metrics mungkin tidak akurat.")
    
    if result.get('error'):
        st.error(f"Clustering memiliki masalah: {result.get('error', 'Unknown error')}")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        delta_text = "Fallback" if result.get('fallback') else "Normal"
        delta_color = "off" if result.get('fallback') else "normal"
        st.metric("Jumlah Cluster", k_value, 
                 delta=delta_text,
                 delta_color=delta_color,
                 help="Jumlah kelompok yang dibuat")
    
    with col2:
        silhouette = result['metrics']['silhouette']
        if silhouette < 0:  # Error value
            st.metric("Silhouette Score", "ERROR", 
                     delta="Check Data", 
                     delta_color="off",
                     help="Tidak dapat dihitung")
        else:
            if silhouette > 0.7:
                quality = "Excellent"
                delta_color = "normal"
            elif silhouette > 0.5:
                quality = "Good"
                delta_color = "normal"
            elif silhouette > 0.3:
                quality = "Fair"
                delta_color = "off"
            else:
                quality = "Poor"
                delta_color = "off"
                
            st.metric("Silhouette Score", f"{silhouette:.3f}", 
                     delta=quality, 
                     delta_color=delta_color,
                     help="Semakin tinggi semakin baik (range: -1 to 1)")
    
    with col3:
        db_score = result['metrics']['davies_bouldin']
        if db_score == float('inf'):
            st.metric("Davies-Bouldin", "ERROR", 
                     delta="Check Data",
                     delta_color="off",
                     help="Tidak dapat dihitung")
        else:
            if db_score < 1:
                quality = "Excellent"
                delta_color = "normal"
            elif db_score < 2:
                quality = "Good"
                delta_color = "normal"
            elif db_score < 3:
                quality = "Fair"
                delta_color = "off"
            else:
                quality = "Poor"
                delta_color = "off"
                
            st.metric("Davies-Bouldin", f"{db_score:.2f}", 
                     delta=quality, 
                     delta_color=delta_color,
                     help="Semakin rendah semakin baik")
    
    with col4:
        inertia = result['metrics']['inertia']
        if inertia == 0:
            st.metric("Inertia", "N/A", 
                     delta="Not calculated",
                     delta_color="off",
                     help="Sum of squared distances")
        else:
            st.metric("Inertia", f"{inertia:,.0f}", 
                     help="Sum of squared distances")
    
    with col5:
        if 'cluster_balance' in result['metrics']:
            balance = result['metrics']['cluster_balance']
            if balance == 0:
                st.metric("Cluster Balance", "N/A",
                         delta="Not calculated",
                         delta_color="off")
            else:
                balance_quality = "Balanced" if balance < 0.5 else "Unbalanced"
                delta_color = "normal" if balance < 0.5 else "off"
                st.metric("Cluster Balance", balance_quality, 
                         delta=f"σ/μ: {balance:.2f}", 
                         delta_color=delta_color)
        else:
            st.metric("Cluster Balance", "N/A",
                     delta="Not available",
                     delta_color="off")
    
    # ==================== TABS ====================
    st.session_state['df_clustered'] = df_clustered
    st.session_state['result'] = result
    st.session_state['k_value'] = k_value
    st.session_state['features_cols'] = features_cols
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "Overview", 
        "Visualisasi", 
        # "Data & Profil",
        "Profiling Kategorikal",
        "Analisis"
    ])
    
    with tab1:
        overview_tab.render(df_clustered, result, k_value, features_cols)
    
    with tab2:
        visualization_tab.render(df_clustered, result, k_value, features_cols)
    
    # with tab3:
    #     data_tab.render(df_clustered, result, k_value, features_cols)

    with tab3:
        categorical_tab.render(df_clustered, result, k_value, features_cols)
    
    with tab4:
        analysis_tab.render(df_clustered, result, k_value, features_cols)
    
    # ==================== FOOTER ====================
    st.markdown("""
    <div style='
        text-align: center;
        padding: 2rem 1rem;
        color: #94A3B8;
        font-size: 0.9rem;
        margin-top: 3rem;
        border-top: 2px solid #E2E8F0;
    '>
        <p style='margin: 0;'>
            <strong>TikTok Content Segmentation Dashboard</strong> | 
            Built with Streamlit & Scikit-learn | Max K = 5
        </p>
        <p style='margin: 0.5rem 0 0 0; font-size: 0.85rem;'>
            © 2024 | Version 2.1 | Logs: tiktok_dashboard.log
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    try:
        main_dashboard()
    except Exception as e:
        logger.critical(f"Critical error in main_dashboard: {str(e)}", exc_info=True)
        st.error(f"""
        Critical Error: {str(e)}
        
        Aplikasi mengalami error yang tidak dapat dipulihkan.
        
        **Langkah troubleshooting:**
        1. Cek file log: tiktok_dashboard.log
        2. Pastikan dataset ada dan formatnya benar
        3. Restart aplikasi
        4. Hubungi developer jika error berlanjut
        """)