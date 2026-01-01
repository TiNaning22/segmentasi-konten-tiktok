import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
import warnings
warnings.filterwarnings('ignore')

# Import tab modules
from tabs import categorical_tab, overview_tab, visualization_tab, data_tab, analysis_tab

# ==================== KONFIGURASI ====================
st.set_page_config(
    page_title="TikTok Content Segmenter",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==================== CUSTOM CSS ====================
def load_css():
    st.markdown("""
    <style>
        /* ===== GLOBAL STYLES ===== */
        .stApp {
            background-color: #F3F7FF;
        }
        
        /* Hide unnecessary elements */
        section[data-testid="stSidebar"],
        header[data-testid="stHeader"],
        #MainMenu,
        footer {
            display: none !important;
        }
        
        /* Main container */
        .main .block-container {
            padding: 1.5rem 2rem 3rem;
            max-width: 100% !important;
        }
        
        /* ===== CARD SYSTEM ===== */
        .custom-card {
            background-color: #FFFFFF;
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
            border: 1px solid #E2E8F0;
            transition: all 0.3s ease;
        }
        
        .custom-card:hover {
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
            transform: translateY(-2px);
        }
        
        .header-card {
            background-color: #FFFFFF;
            border-radius: 12px;
            padding: 2rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
            border: 1px solid #E2E8F0;
        }
        
        .info-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1rem;
        }
        
        .info-item {
            padding: 0.8rem;
            background-color: #F8FAFC;
            border-radius: 8px;
            border: 1px solid #E2E8F0;
            transition: all 0.3s ease;
        }
        
        .info-item:hover {
            background-color: #EEF2FF;
            border-color: #C7D2FE;
        }
        
        .info-label {
            color: #64748B;
            font-size: 0.85rem;
            font-weight: 500;
            margin: 0 0 0.3rem 0;
        }
        
        .info-value {
            color: #1E293B;
            font-size: 1.5rem;
            font-weight: 700;
            margin: 0;
        }
        
        .info-value.success {
            color: #10B981;
        }
        
        .info-value.error {
            color: #EF4444;
        }
        
        /* ===== METRICS ===== */
        [data-testid="metric-container"] {
            background: #FFFFFF;
            border-radius: 12px;
            padding: 1.2rem;
            border: 2px solid #E2E8F0;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
            transition: all 0.3s ease;
        }
        
        [data-testid="metric-container"]:hover {
            border-color: #3B82F6;
            box-shadow: 0 4px 16px rgba(59, 130, 246, 0.15);
            transform: translateY(-2px);
        }
        
        [data-testid="stMetricLabel"] {
            color: #64748B !important;
            font-size: 0.9rem !important;
            font-weight: 500 !important;
        }
        
        [data-testid="stMetricValue"] {
            color: #1E293B !important;
            font-size: 1.8rem !important;
            font-weight: 700 !important;
        }
        
        [data-testid="stMetricDelta"] {
            font-size: 0.85rem !important;
        }
        
        /* ===== TYPOGRAPHY ===== */
        h1, h2, h3, h4 {
            color: #1E293B;
            font-weight: 700;
            letter-spacing: -0.5px;
        }
        
        h1 { font-size: 2rem; }
        h2 { font-size: 1.6rem; }
        h3 { font-size: 1.3rem; }
        
        p {
            line-height: 1.6;
            color: #475569;
        }
        
        /* ===== TABS ===== */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
            background: #FFFFFF;
            border-radius: 12px;
            padding: 0.5rem;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
            border: 1px solid #E2E8F0;
        }
        
        .stTabs [data-baseweb="tab"] {
            background: transparent;
            color: #64748B;
            border-radius: 8px;
            padding: 0.6rem 1.5rem;
            border: none;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            background: #F1F5F9;
            color: #475569;
        }
        
        .stTabs [aria-selected="true"] {
            background: #3B82F6;
            color: white !important;
            box-shadow: 0 2px 8px rgba(59, 130, 246, 0.3);
        }
        
        /* ===== BUTTONS ===== */
        .stButton > button {
            background-color: #3B82F6;
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.6rem 1.8rem;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 2px 8px rgba(59, 130, 246, 0.3);
        }
        
        .stButton > button:hover {
            background-color: #2563EB;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
        }
        
        .stDownloadButton > button {
            background-color: #10B981;
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.6rem 1.8rem;
            font-weight: 600;
            box-shadow: 0 2px 8px rgba(16, 185, 129, 0.3);
        }
        
        .stDownloadButton > button:hover {
            background-color: #059669;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(16, 185, 129, 0.4);
        }
        
        /* ===== INPUTS ===== */
        .stSlider > div > div > div {
            background-color: #3B82F6;
        }
        
        .stSelectbox > div > div,
        .stMultiSelect > div > div {
            background: #FFFFFF;
            border: 2px solid #E2E8F0;
            border-radius: 10px;
            transition: all 0.3s ease;
        }
        
        .stSelectbox > div > div:hover,
        .stMultiSelect > div > div:hover {
            border-color: #3B82F6;
        }
        
        /* ===== DATAFRAME ===== */
        .dataframe {
            background: #FFFFFF;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        }
        
        .dataframe thead tr th {
            background-color: #3B82F6;
            color: white !important;
            font-weight: 600;
            padding: 12px;
        }
        
        .dataframe tbody tr:hover {
            background: #F8FAFC;
        }
        
        /* ===== ALERTS ===== */
        .stAlert {
            border-radius: 12px;
            border: none;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        }
        
        /* ===== LOADING ===== */
        .stSpinner > div {
            border-color: #3B82F6 transparent #3B82F6 transparent !important;
        }
        
        /* ===== PLOTLY CHARTS ===== */
        .js-plotly-plot {
            border-radius: 12px;
            overflow: hidden;
        }
        
        /* ===== SCROLLBAR ===== */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: #F1F5F9;
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: #3B82F6;
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: #2563EB;
        }
        
        /* ===== RESPONSIVE ===== */
        @media (max-width: 768px) {
            .main .block-container {
                padding: 1rem;
            }
            
            .header-card {
                padding: 1.5rem;
            }
            
            .info-grid {
                grid-template-columns: 1fr;
            }
            
            h1 { font-size: 1.6rem; }
            h2 { font-size: 1.4rem; }
            h3 { font-size: 1.2rem; }
        }
        
        /* ===== ANIMATIONS ===== */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .custom-card, [data-testid="metric-container"] {
            animation: fadeIn 0.5s ease-out;
        }
    </style>
    """, unsafe_allow_html=True)

# ==================== FUNGSI HELPER ====================
@st.cache_data
def load_data():
    """Load dataset TikTok dengan preprocessing lengkap"""
    try:
        df = pd.read_csv('tiktok_digital_marketing_data.csv')
    except FileNotFoundError:
        st.error("❌ Dataset tidak ditemukan. Pastikan 'tiktok_digital_marketing_data.csv' ada di direktori.")
        st.stop()
    
    # Validate required columns
    required_cols = ['Likes', 'Shares', 'Comments', 'Views', 'TimeSpentOnContent']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"❌ Kolom yang hilang: {', '.join(missing_cols)}")
        st.stop()
    
    numeric_cols = ['Likes', 'Shares', 'Comments', 'Views', 'TimeSpentOnContent']
    
    # Handle missing values
    missing_count = df[numeric_cols].isnull().sum().sum()
    if missing_count > 0:
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        st.info(f"ℹ️ Mengisi {missing_count} nilai yang hilang dengan median")
    
    # Calculate engagement rate (avoid division by zero)
    df['Engagement_Rate'] = (
        (df['Likes'] + df['Comments'] + df['Shares']) / 
        df['Views'].clip(lower=1)
    )
    
    # Sample if needed for performance
    MAX_ROWS = 100000
    original_size = len(df)
    if len(df) > MAX_ROWS:
        df = df.sample(n=MAX_ROWS, random_state=42).reset_index(drop=True)
        st.warning(f"⚠️ Dataset disampling dari {original_size:,} ke {MAX_ROWS:,} baris untuk performa optimal")
    
    return df

@st.cache_data
def perform_clustering(df, n_clusters, features_cols):
    """Perform K-Means clustering"""
    # Validate inputs
    if n_clusters < 2:
        raise ValueError("n_clusters harus >= 2")
    
    if len(df) < 10:
        raise ValueError("Dataset terlalu kecil untuk clustering (minimum 10 baris)")
    
    features = df[features_cols].copy()
    
    # Check for infinite values
    if not np.isfinite(features).all().all():
        st.warning("⚠️ Mendeteksi nilai infinite, diganti dengan median")
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(features.median())
    
    # Standardization
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # K-Means clustering
    kmeans = KMeans(
        n_clusters=n_clusters, 
        random_state=42, 
        n_init=10, 
        max_iter=300
    )
    clusters = kmeans.fit_predict(scaled_features)
    
    # Calculate metrics (with sampling for large datasets)
    sample_size = min(5000, len(scaled_features))
    if len(scaled_features) > sample_size:
        indices = np.random.choice(len(scaled_features), sample_size, replace=False)
        silhouette = silhouette_score(scaled_features[indices], clusters[indices])
    else:
        silhouette = silhouette_score(scaled_features, clusters)
    
    davies_bouldin = davies_bouldin_score(scaled_features, clusters)
    inertia = kmeans.inertia_
    
    # PCA for visualization
    if len(scaled_features) > 5000:
        sample_idx = np.random.choice(len(scaled_features), 5000, replace=False)
        pca = PCA(n_components=2)
        pca_result_sample = pca.fit_transform(scaled_features[sample_idx])
        pca_result = np.zeros((len(scaled_features), 2))
        pca_result[sample_idx] = pca_result_sample
        pca_explained = pca.explained_variance_ratio_
        use_sample = True
        sample_indices = sample_idx
    else:
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(scaled_features)
        pca_explained = pca.explained_variance_ratio_
        use_sample = False
        sample_indices = None
    
    # Cluster sizes for balance metric
    cluster_sizes = np.bincount(clusters)
    
    return {
        'clusters': clusters,
        'kmeans': kmeans,
        'scaler': scaler,
        'scaled_features': scaled_features,
        'pca_result': pca_result,
        'pca_explained': pca_explained,
        'use_sample': use_sample,
        'sample_indices': sample_indices,
        'metrics': {
            'silhouette': silhouette,
            'davies_bouldin': davies_bouldin,
            'inertia': inertia,
            'cluster_sizes': cluster_sizes
        }
    }

# ==================== DASHBOARD UTAMA ====================
def main_dashboard():
    # Load CSS
    load_css()
    
    # ==================== HEADER ====================
    st.markdown("""
    <div class='header-card'>
        <h1 style='color: #1E293B; margin: 0 0 0.5rem 0;'>
            📊 TikTok Intelligence Dashboard
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
                🎯 Kontrol Clustering
            </h3>
        </div>
        """, unsafe_allow_html=True)
        
        k_value = st.slider(
            "Jumlah Cluster (K):",
            min_value=2,
            max_value=10,
            value=4,
            help="Jumlah kelompok konten yang ingin dibuat"
        )
    
    with col2:
        df = load_data()
        features_cols = ['Likes', 'Shares', 'Comments', 'Views', 'TimeSpentOnContent', 'Engagement_Rate']
        
        missing_values = df[features_cols].isnull().sum().sum()
        
        st.markdown(f"""
        <div class='custom-card'>
            <h3 style='color: #1E293B; margin: 0 0 1rem 0;'>
                📊 Dataset Information
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
                    <p class='info-value {"success" if missing_values == 0 else "error"}'>{missing_values}</p>
                </div>
                <div class='info-item'>
                    <p class='info-label'>Engagement Rate</p>
                    <p class='info-value success'>✓ Aktif</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ==================== CLUSTERING ====================
    with st.spinner("🔄 Melakukan clustering..."):
        try:
            result = perform_clustering(df, k_value, features_cols)
        except Exception as e:
            st.error(f"❌ Error dalam clustering: {str(e)}")
            st.stop()
    
    df_clustered = df.copy()
    df_clustered['Cluster'] = result['clusters']
    
    # ==================== METRICS ====================
    st.markdown("""
    <div class='custom-card'>
        <h3 style='color: #1E293B; margin: 0 0 1.5rem 0;'>
            📈 Model Performance Metrics
        </h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Jumlah Cluster", k_value, help="Jumlah kelompok yang dibuat")
    
    with col2:
        silhouette = result['metrics']['silhouette']
        quality = "Excellent" if silhouette > 0.7 else "Good" if silhouette > 0.5 else "Fair" if silhouette > 0.3 else "Poor"
        st.metric("Silhouette Score", f"{silhouette:.3f}", delta=quality, help="Semakin tinggi semakin baik (range: -1 to 1)")
    
    with col3:
        db_score = result['metrics']['davies_bouldin']
        quality = "Excellent" if db_score < 1 else "Good" if db_score < 2 else "Fair"
        st.metric("Davies-Bouldin", f"{db_score:.2f}", delta=quality, delta_color="inverse", help="Semakin rendah semakin baik")
    
    with col4:
        st.metric("Inertia", f"{result['metrics']['inertia']:,.0f}", help="Sum of squared distances")
    
    with col5:
        cluster_sizes = result['metrics']['cluster_sizes']
        balance = np.std(cluster_sizes) / np.mean(cluster_sizes)
        balance_quality = "Balanced" if balance < 0.5 else "Unbalanced"
        st.metric("Cluster Balance", balance_quality, delta=f"σ/μ: {balance:.2f}", delta_color="normal" if balance < 0.5 else "inverse")
    
    # ==================== TABS ====================
    st.session_state['df_clustered'] = df_clustered
    st.session_state['result'] = result
    st.session_state['k_value'] = k_value
    st.session_state['features_cols'] = features_cols
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", 
        "🔍 Visualisasi", 
        "📋 Data & Profil",
        "👥 Profiling Kategorikal",
        "📈 Analisis"
    ])
    
    with tab1:
        overview_tab.render(df_clustered, result, k_value, features_cols)
    
    with tab2:
        visualization_tab.render(df_clustered, result, k_value, features_cols)
    
    with tab3:
        data_tab.render(df_clustered, result, k_value, features_cols)

    with tab4:
        categorical_tab.render(df_clustered, result, k_value, features_cols)
    
    with tab5:
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
            Built with Streamlit & Scikit-learn
        </p>
        <p style='margin: 0.5rem 0 0 0; font-size: 0.85rem;'>
            © 2024 | Version 2.0
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main_dashboard()