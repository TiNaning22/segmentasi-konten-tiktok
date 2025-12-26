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
        /* Background utama - F3F7FF */
        .stApp {
            background-color: #F3F7FF;
            background-image: none !important;
        }
        
        /* Hide sidebar */
        section[data-testid="stSidebar"] {
            display: none !important;
        }
        
        /* Hide default header/footer */
        header[data-testid="stHeader"] {
            display: none !important;
        }
        
        /* Main content container */
        .main .block-container {
            padding-top: 1rem;
            padding-left: 2rem;
            padding-right: 2rem;
            max-width: 100% !important;
        }
        
        /* Card styling - putih tanpa border */
        .custom-card {
            background-color: #FFFFFF;
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
            border: none !important;
        }
        
        /* Metric cards - putih tanpa border */
        [data-testid="metric-container"] {
            background-color: #FFFFFF;
            border-radius: 10px;
            padding: 1rem;
            border: none;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        }
        
        /* Metric label - warna hitam */
        [data-testid="stMetricLabel"] {
            color: #1E293B !important;
        }
        
        /* Metric value - warna hitam */
        [data-testid="stMetricValue"] {
            color: #1E293B !important;
        }
        
        /* Metric delta */
        [data-testid="stMetricDelta"] {
            color: #64748B !important;
        }
        
        /* Headers */
        h1, h2, h3, h4 {
            color: #1E293B;
            font-weight: 600;
            margin-top: 0;
        }
        
        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 1rem;
            background-color: transparent;
            border-bottom: 2px solid #E2E8F0;
        }
        
        .stTabs [data-baseweb="tab"] {
            background-color: transparent;
            color: #64748B;
            border-radius: 6px 6px 0 0;
            padding: 0.5rem 1.5rem;
            border: none;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: #FFFFFF;
            color: #3B82F6;
            border-bottom: 3px solid #3B82F6;
        }
        
        /* Button styling */
        .stButton > button {
            background-color: #3B82F6;
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.5rem 1.5rem;
            font-weight: 500;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            background-color: #2563EB;
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.2);
        }
        
        /* Download button */
        .stDownloadButton > button {
            background-color: #10B981;
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.5rem 1.5rem;
        }
        
        .stDownloadButton > button:hover {
            background-color: #059669;
        }
        
        /* Slider styling */
        .stSlider > div > div {
            background-color: #3B82F6;
        }
        
        /* Selectbox/Dropdown */
        .stSelectbox > div > div {
            background-color: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 8px;
        }
        
        /* Dataframe styling */
        .dataframe {
            background-color: #FFFFFF;
            border-radius: 8px;
            overflow: hidden;
        }
        
        /* Success, Warning, Info boxes */
        .stAlert {
            border-radius: 8px;
            border: none;
            background-color: #FFFFFF;
        }
        
        /* Hide hamburger menu */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Loading spinner */
        .stSpinner > div {
            border-color: #3B82F6 !important;
        }
        
        /* Responsive design */
        @media (max-width: 768px) {
            .main .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }
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
    MAX_ROWS = 10000
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
    
    # ==================== HEADER CARD ====================
    st.markdown("""
    <div style='
        background-color: #FFFFFF;
        border-radius: 12px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
        border: none;
    '>
        <h1 style='color: #1E293B; margin: 0 0 0.5rem 0; font-size: 2rem;'>
            TikTok Intelligence Dashboard
        </h1>
        <p style='color: #64748B; font-size: 1rem; margin: 0;'>
            Segmentasi Konten Otomatis Menggunakan K-Means
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # ==================== KONTROL & INFO CARDS ====================
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Control panel card
        st.markdown("""
        <div style='
            background-color: #FFFFFF;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
            border: none;
        '>
            <h3 style='color: #1E293B; margin: 0 0 1rem 0; font-size: 1.1rem;'>
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
        # Dataset info card
        df = load_data()
        features_cols = ['Likes', 'Shares', 'Comments', 'Views', 'TimeSpentOnContent', 'Engagement_Rate']
        
        st.markdown(f"""
        <div style='
            background-color: #FFFFFF;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
            border: none;
        '>
            <h3 style='color: #1E293B; margin: 0 0 1rem 0; font-size: 1.1rem;'>
                📊 Dataset Info
            </h3>
            <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 0.8rem;'>
                <div>
                    <p style='color: #64748B; margin: 0; font-size: 0.85rem;'>Total Konten</p>
                    <p style='color: #1E293B; margin: 0; font-size: 1.3rem; font-weight: 600;'>{len(df):,}</p>
                </div>
                <div>
                    <p style='color: #64748B; margin: 0; font-size: 0.85rem;'>Features</p>
                    <p style='color: #1E293B; margin: 0; font-size: 1.3rem; font-weight: 600;'>{len(features_cols)}</p>
                </div>
                <div>
                    <p style='color: #64748B; margin: 0; font-size: 0.85rem;'>Missing Values</p>
                    <p style='color: #1E293B; margin: 0; font-size: 1.3rem; font-weight: 600;'>{df[features_cols].isnull().sum().sum()}</p>
                </div>
                <div>
                    <p style='color: #64748B; margin: 0; font-size: 0.85rem;'>Engagement Rate</p>
                    <p style='color: #10B981; margin: 0; font-size: 1.3rem; font-weight: 600;'>✓ Aktif</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ==================== CLUSTERING ====================
    # Perform clustering
    with st.spinner("🔄 Melakukan clustering..."):
        try:
            result = perform_clustering(df, k_value, features_cols)
        except Exception as e:
            st.error(f"❌ Error dalam clustering: {str(e)}")
            st.stop()
    
    df_clustered = df.copy()
    df_clustered['Cluster'] = result['clusters']
    
    # ==================== METRICS CARD ====================
    st.markdown("""
    <div style='
        background-color: #FFFFFF;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
        border: none;
        margin-bottom: 1.5rem;
    '>
        <h3 style='color: #1E293B; margin: 0 0 1.5rem 0; font-size: 1.1rem;'>
            📈 Model Performance Metrics
        </h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Metrics dalam columns - native streamlit (sudah ada card dari CSS)
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Jumlah Cluster",
            k_value,
            help="Jumlah kelompok yang dibuat"
        )
    
    with col2:
        silhouette = result['metrics']['silhouette']
        quality = "Excellent" if silhouette > 0.7 else "Good" if silhouette > 0.5 else "Fair" if silhouette > 0.3 else "Poor"
        st.metric(
            "Silhouette Score",
            f"{silhouette:.3f}",
            delta=quality,
            help="Semakin tinggi semakin baik (range: -1 to 1)"
        )
    
    with col3:
        db_score = result['metrics']['davies_bouldin']
        quality = "Excellent" if db_score < 1 else "Good" if db_score < 2 else "Fair"
        st.metric(
            "Davies-Bouldin",
            f"{db_score:.2f}",
            delta=quality,
            delta_color="inverse",
            help="Semakin rendah semakin baik"
        )
    
    with col4:
        st.metric(
            "Inertia",
            f"{result['metrics']['inertia']:,.0f}",
            help="Sum of squared distances"
        )
    
    with col5:
        cluster_sizes = result['metrics']['cluster_sizes']
        balance = np.std(cluster_sizes) / np.mean(cluster_sizes)
        balance_quality = "Balanced" if balance < 0.5 else "Unbalanced"
        balance_color = "normal" if balance < 0.5 else "inverse"
        st.metric(
            "Cluster Balance",
            balance_quality,
            delta=f"σ/μ: {balance:.2f}",
            delta_color=balance_color,
            help="Keseimbangan distribusi cluster"
        )
    
    # ==================== TABS ====================
    # Simpan data di session state untuk diakses tab
    st.session_state['df_clustered'] = df_clustered
    st.session_state['result'] = result
    st.session_state['k_value'] = k_value
    st.session_state['features_cols'] = features_cols
    
    # Tab utama
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
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='
        text-align: center;
        padding: 1.5rem;
        color: #94A3B8;
        font-size: 13px;
    '>
        © 2024 TikTok Content Segmentation Dashboard | Built with Streamlit & Scikit-learn
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main_dashboard()