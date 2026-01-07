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
    page_icon="",
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
    
    try:
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
            
            # Load data untuk mendapatkan available features
            df_loaded = False
            try:
                df = load_data()
                df_loaded = True
                
                # Deteksi kolom numerik yang tersedia
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                default_features = ['Likes', 'Shares', 'Comments', 'Views', 
                                   'TimeSpentOnContent', 'Engagement_Rate']
                
                # Filter hanya yang ada di dataset
                available_features = [col for col in default_features if col in df.columns and col in numeric_cols]
                
                # Tambahkan kolom numerik lainnya
                additional_numeric = [col for col in numeric_cols 
                                     if col not in available_features 
                                     and col != 'Cluster'
                                     and pd.api.types.is_numeric_dtype(df[col])]
                
                all_possible_features = available_features + additional_numeric[:10]
                
            except Exception as e:
                logger.error(f"Error loading data: {str(e)}")
                df_loaded = False
                all_possible_features = ['Likes', 'Shares', 'Comments', 'Views', 
                                        'TimeSpentOnContent', 'Engagement_Rate']
                default_features = all_possible_features
            
            # FEATURE SELECTION MULTI-SELECT
            st.markdown("**Pilih Features untuk Clustering:**")
            selected_features = st.multiselect(
                "Features:",
                options=all_possible_features,
                default=default_features[:4] if len(default_features) >= 4 else default_features,
                help="Pilih minimal 2 features untuk clustering"
            )
            
            # Validasi features yang dipilih
            valid_features = []
            if df_loaded:
                for feature in selected_features:
                    if feature in df.columns:
                        if pd.api.types.is_numeric_dtype(df[feature]):
                            valid_features.append(feature)
                        else:
                            st.warning(f"Feature '{feature}' bukan numerik, diabaikan")
                    else:
                        st.warning(f"Feature '{feature}' tidak ditemukan, diabaikan")
            else:
                valid_features = selected_features
            
            # Pastikan minimal 2 features valid
            if len(valid_features) < 2:
                st.error(f"Hanya {len(valid_features)} features valid. Pilih minimal 2 features numerik.")
                if df_loaded:
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    st.info(f"Features numerik yang tersedia: {', '.join(numeric_cols[:10])}{'...' if len(numeric_cols) > 10 else ''}")
                st.stop()
            
            features_cols = valid_features
            
            # K value slider
            if df_loaded:
                suggestions = suggest_optimal_clusters(df, features_cols)
            else:
                suggestions = {'recommended': 4, 'recommended_range': '2-5'}
            
            default_k = min(suggestions.get('recommended', 4), 5)
            
            k_value = st.slider(
                "Jumlah Cluster (K):",
                min_value=2,
                max_value=5,
                value=default_k,
                help=f"Disarankan: {suggestions.get('recommended_range', '2-5')}. K={default_k} berdasarkan data"
            )
        
        with col2:
            if not df_loaded:
                st.error("Tidak dapat memuat data. Cek file dataset.")
                st.stop()
            
            df = load_data()
            
            missing_values = df[features_cols].isna().sum().sum()
            
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
                        <p class='info-label'>Features Selected</p>
                        <p class='info-value'>{len(features_cols)}</p>
                    </div>
                    <div class='info-item'>
                        <p class='info-label'>Missing Values</p>
                        <p class='info-value {'success' if missing_values == 0 else 'warning' if missing_values < 10 else 'error'}">
                            {missing_values}
                        </p>
                    </div>
                    <div class='info-item'>
                        <p class='info-label'>Data Size</p>
                        <p class='info-value {'success' if len(df) <= 10000 else 'warning' if len(df) <= 50000 else 'error'}">
                            {'Small' if len(df) <= 10000 else 'Medium' if len(df) <= 50000 else 'Large'}
                        </p>
                    </div>
                </div>
                <div style='margin-top: 1rem; font-size: 0.85rem; color: #FFFFFF;'>
                    <strong>Features:</strong> {', '.join(features_cols[:5])}{'...' if len(features_cols) > 5 else ''}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Quick data validation
            is_valid, validation_msg, validation_warnings = validate_data_for_clustering(df, features_cols)
            
            if not is_valid:
                st.error(validation_msg)
                st.stop()
            else:
                if validation_warnings:
                    with st.expander("Validation Warnings", expanded=False):
                        for warning in validation_warnings[:3]:
                            st.warning(warning)
        
        # ==================== CLUSTERING ====================
        st.markdown("<br>", unsafe_allow_html=True)
        
        with st.spinner("Melakukan clustering..."):
            progress_bar = st.progress(0)
            
            try:
                progress_bar.progress(20)
                
                # Validasi data
                is_valid, validation_msg, validation_warnings = validate_data_for_clustering(df, features_cols)
                
                if not is_valid:
                    progress_bar.progress(100)
                    st.error(validation_msg)
                    st.stop()
                
                progress_bar.progress(40)
                
                # Lakukan clustering
                result = perform_clustering(df, k_value, features_cols)
                
                progress_bar.progress(80)
                
                if result and not result.get('success', True):
                    st.error(f"Error dalam clustering: {result.get('error', 'Unknown error')}")
                    
                    if k_value > 2:
                        st.warning(f"Mencoba clustering dengan K={k_value-1}...")
                        result = perform_clustering(df, k_value-1, features_cols)
                        
                        if not result.get('success', True):
                            st.error("Clustering tetap gagal. Silakan cek data Anda.")
                            st.stop()
                        else:
                            st.success(f"Clustering berhasil dengan K={k_value-1}")
                            k_value = k_value - 1
                    else:
                        st.error("Tidak dapat melakukan clustering. Cek data dan coba lagi.")
                        st.stop()
                
                progress_bar.progress(100)
                
            except Exception as e:
                progress_bar.progress(100)
                logger.error(f"Exception tidak terduga: {str(e)}", exc_info=True)
                st.error(f"Exception tidak terduga: {str(e)}")
                
                # Fallback sederhana
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
        
        # ==================== TABS ====================
        st.session_state['df_clustered'] = df_clustered
        st.session_state['result'] = result
        st.session_state['k_value'] = k_value
        st.session_state['features_cols'] = features_cols
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "Overview", 
            "Visualisasi", 
            "Profiling Kategorikal",
            "Analisis"
        ])
        
        with tab1:
            overview_tab.render(df_clustered, result, k_value, features_cols)
        
        with tab2:
            visualization_tab.render(df_clustered, result, k_value, features_cols)
        
        with tab3:
            categorical_tab.render(df_clustered, result, k_value, features_cols)
        
        with tab4:
            analysis_tab.render(df_clustered, result, k_value, features_cols)
    
    except Exception as e:
        logger.critical(f"Critical error in main_dashboard: {str(e)}", exc_info=True)
        st.error(f"""
        ## Critical Error
        
        **Error:** {str(e)}
        
        **Langkah troubleshooting:**
        1. Cek file log: tiktok_dashboard.log
        2. Pastikan dataset ada dan formatnya benar
        3. Kurangi jumlah features yang dipilih
        4. Restart aplikasi
        
        **Tips cepat:**
        - Pilih hanya 2-3 features untuk testing
        - Coba dataset demo dengan tombol di bawah
        """)
        
        if st.button("Generate Dataset Demo"):
            np.random.seed(42)
            n_samples = 1000
            demo_data = pd.DataFrame({
                'Likes': np.random.randint(100, 10000, n_samples),
                'Shares': np.random.randint(10, 1000, n_samples),
                'Comments': np.random.randint(5, 500, n_samples),
                'Views': np.random.randint(1000, 100000, n_samples),
                'TimeSpentOnContent': np.random.uniform(10, 300, n_samples),
                'ContentType': np.random.choice(['Video', 'Image', 'Text'], n_samples),
                'AgeGroup': np.random.choice(['18-24', '25-34', '35-44'], n_samples),
                'Location': np.random.choice(['Jakarta', 'Surabaya', 'Bandung', 'Medan'], n_samples)
            })
            
            demo_data['Engagement_Rate'] = (
                (demo_data['Likes'] + demo_data['Comments'] + demo_data['Shares']) / 
                demo_data['Views'].clip(lower=1)
            )
            
            demo_data.to_csv('tiktok_demo_data.csv', index=False)
            st.success("Data demo berhasil dibuat. Silakan refresh halaman.")
            st.stop()
    
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