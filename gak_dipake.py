import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
import warnings
import logging
from typing import Dict, Any, Optional, Tuple, List
import traceback

# Import tab modules
from tabs import categorical_tab, overview_tab, visualization_tab, data_tab, analysis_tab

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
    
    .info-value.warning {
        color: #F59E0B;
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
    
    /* Error alert styling */
    .stAlert[data-testid="stAlert"] div[role="alert"] {
        border-left: 4px solid !important;
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
    
    /* ===== DIAGNOSTIC STYLES ===== */
    .diagnostic-card {
        background: linear-gradient(135deg, #F8FAFC 0%, #F1F5F9 100%);
        border-left: 4px solid #3B82F6;
    }
    
    .warning-card {
        background: linear-gradient(135deg, #FEF3C7 0%, #FDE68A 100%);
        border-left: 4px solid #F59E0B;
    }
    
    .error-card {
        background: linear-gradient(135deg, #FEE2E2 0%, #FECACA 100%);
        border-left: 4px solid #EF4444;
    }
    
    .success-card {
        background: linear-gradient(135deg, #D1FAE5 0%, #A7F3D0 100%);
        border-left: 4px solid #10B981;
    }
    
    .tip-box {
        background: linear-gradient(135deg, #F0F9FF 0%, #E0F2FE 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #0EA5E9;
        margin: 1rem 0;
    }
    
    /* ===== PROGRESS BAR ===== */
    .stProgress > div > div > div > div {
        background-color: #3B82F6;
    }
    </style>
    """, unsafe_allow_html=True)

# ==================== FUNGSI HELPER ====================
def validate_data_for_clustering(df: pd.DataFrame, features_cols: list) -> Tuple[bool, str, List[str]]:
    """
    Validasi dataset untuk clustering
    
    Returns:
    --------
    Tuple[bool, str, List[str]]: (is_valid, message, warnings)
    """
    
    warnings = []
    
    # 1. Cek dataframe tidak kosong
    if df.empty:
        return False, "❌ DataFrame kosong", []
    
    # 2. Cek minimal rows
    if len(df) < 10:
        return False, f"❌ Data terlalu sedikit ({len(df)} rows). Minimal 10 rows", []
    
    # 3. Cek features exist
    missing = [col for col in features_cols if col not in df.columns]
    if missing:
        return False, f"❌ Kolom tidak ditemukan: {missing}", []
    
    # 4. Cek tipe data numerik
    numeric_cols = df[features_cols].select_dtypes(include=[np.number]).columns
    non_numeric = [col for col in features_cols if col not in numeric_cols]
    if non_numeric:
        return False, f"❌ Kolom non-numerik: {non_numeric}", []
    
    # 5. Cek missing values percentage
    missing_pct = (df[features_cols].isnull().sum().sum() / 
                  (len(df) * len(features_cols))) * 100
    if missing_pct > 50:
        return False, f"❌ Missing values terlalu tinggi ({missing_pct:.1f}%)", []
    elif missing_pct > 10:
        warnings.append(f"Missing values: {missing_pct:.1f}%")
    
    # 6. Cek zero variance
    variances = df[features_cols].var()
    zero_var = variances[variances == 0].index.tolist()
    if len(zero_var) == len(features_cols):
        return False, "❌ Semua features memiliki zero variance", []
    elif zero_var:
        warnings.append(f"Zero variance features: {zero_var}")
    
    # 7. Cek outliers ekstrem (optional warning)
    extreme_outlier_cols = []
    for col in features_cols:
        if col in df.columns:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                lower_bound = q1 - 10 * iqr
                upper_bound = q3 + 10 * iqr
                extreme_outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                if extreme_outliers > 0:
                    extreme_outlier_cols.append(f"{col}: {extreme_outliers}")
    
    if extreme_outlier_cols:
        warnings.append(f"Extreme outliers detected in: {', '.join(extreme_outlier_cols)}")
    
    # 8. Cek korelasi sangat tinggi antar features
    if len(features_cols) > 1:
        corr_matrix = df[features_cols].corr().abs()
        np.fill_diagonal(corr_matrix.values, 0)
        high_corr_pairs = []
        for i in range(len(features_cols)):
            for j in range(i+1, len(features_cols)):
                if corr_matrix.iloc[i, j] > 0.95:
                    high_corr_pairs.append(f"{features_cols[i]}-{features_cols[j]}: {corr_matrix.iloc[i, j]:.2f}")
        
        if high_corr_pairs:
            warnings.append(f"High correlation (>0.95): {', '.join(high_corr_pairs[:3])}")
    
    message = "✅ Data valid untuk clustering"
    if warnings:
        message += f" ({len(warnings)} warnings)"
    
    return True, message, warnings

def suggest_optimal_clusters(df: pd.DataFrame, features_cols: list, max_k: int = 10) -> Dict:
    """
    Berikan saran jumlah cluster optimal
    """
    suggestions = {
        'based_on_size': min(10, max(2, len(df) // 100)),
        'based_on_features': min(8, max(2, len(features_cols) * 2)),
        'recommended_range': '',
        'warnings': []
    }
    
    if len(df) < 50:
        suggestions['based_on_size'] = min(5, max(2, len(df) // 10))
        suggestions['warnings'].append(f"Data kecil ({len(df)} rows)")
    
    if len(features_cols) < 3:
        suggestions['warnings'].append(f"Hanya {len(features_cols)} features")
    
    # Jika data cukup besar, berikan range
    if len(df) > 100:
        min_k = max(2, len(features_cols))
        max_k = min(8, len(df) // 50)
        if min_k <= max_k:
            suggestions['recommended_range'] = f"{min_k}-{max_k}"
        else:
            suggestions['recommended_range'] = f"2-{max_k}"
    
    # Pilih nilai yang paling konservatif
    suggestions['recommended'] = min(
        suggestions['based_on_size'],
        suggestions['based_on_features']
    )
    
    return suggestions

def get_user_friendly_error(error_type: str, details: str = "") -> str:
    """
    Convert technical errors to user-friendly messages
    """
    error_messages = {
        'MEMORY_ERROR': "Data terlalu besar. Coba sampling data atau kurangi features.",
        'VARIANCE_ERROR': "Features tidak memiliki variasi. Coba pilih features yang berbeda.",
        'CLUSTER_ERROR': "Tidak dapat membentuk cluster. Coba kurangi jumlah cluster.",
        'CONVERGENCE_ERROR': "Algoritma tidak konvergen. Coba tingkatkan max_iter atau ubah random_state.",
        'SCALING_ERROR': "Error dalam normalisasi data. Cek outliers atau missing values.",
        'PCA_ERROR': "Tidak dapat membuat visualisasi. Lanjutkan tanpa PCA.",
        'DATA_TOO_SMALL': "Data terlalu sedikit untuk clustering. Minimal 10 data points.",
        'INVALID_FEATURES': "Features tidak valid. Pastikan semua features numerik.",
        'MISSING_VALUES': "Terlalu banyak missing values. Coba imputasi atau hapus rows.",
        'ZERO_VARIANCE': "Semua data sama untuk beberapa features. Tidak ada variasi untuk clustering.",
    }
    
    base_message = error_messages.get(error_type, "Terjadi kesalahan yang tidak diketahui.")
    
    if details:
        return f"{base_message} Detail: {details}"
    
    return base_message

@st.cache_data
def load_data():
    """Load dataset TikTok dengan preprocessing lengkap"""
    logger.info("Memulai loading data...")
    
    try:
        df = pd.read_csv('tiktok_digital_marketing_data.csv')
        logger.info(f"Data berhasil dimuat. Shape: {df.shape}")
        
    except FileNotFoundError:
        logger.error("File dataset tidak ditemukan")
        st.error("""
        ❌ Dataset tidak ditemukan. 
        
        Pastikan file 'tiktok_digital_marketing_data.csv' ada di direktori yang sama.
        
        **Alternatif:** Gunakan data demo untuk testing:
        """)
        
        # Create demo data
        if st.button("Generate Data Demo"):
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
            
            # Calculate engagement rate
            demo_data['Engagement_Rate'] = (
                (demo_data['Likes'] + demo_data['Comments'] + demo_data['Shares']) / 
                demo_data['Views'].clip(lower=1)
            )
            
            # Save demo data
            demo_data.to_csv('tiktok_demo_data.csv', index=False)
            st.success("✅ Data demo berhasil dibuat. Silakan refresh halaman.")
            st.stop()
        
        st.stop()
    
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        st.error(f"❌ Error membaca file CSV: {str(e)}")
        st.stop()
    
    # Validate required columns
    required_cols = ['Likes', 'Shares', 'Comments', 'Views', 'TimeSpentOnContent']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        st.error(f"""
        ❌ Kolom yang hilang: {', '.join(missing_cols)}
        
        **Kolom yang dibutuhkan:**
        - Likes
        - Shares  
        - Comments
        - Views
        - TimeSpentOnContent
        
        Pastikan dataset Anda memiliki kolom-kolom tersebut.
        """)
        st.stop()
    
    numeric_cols = ['Likes', 'Shares', 'Comments', 'Views', 'TimeSpentOnContent']
    
    # Handle missing values
    missing_count = df[numeric_cols].isnull().sum().sum()
    if missing_count > 0:
        logger.warning(f"Mengisi {missing_count} missing values dengan median")
        
        # Fill with median per column
        for col in numeric_cols:
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                logger.info(f"Kolom {col}: mengisi {df[col].isnull().sum()} values dengan median {median_val:.2f}")
        
        st.info(f"ℹ️ Mengisi {missing_count} nilai yang hilang dengan median")
    
    # Calculate engagement rate (avoid division by zero)
    df['Engagement_Rate'] = (
        (df['Likes'] + df['Comments'] + df['Shares']) / 
        df['Views'].clip(lower=1)
    )
    
    # Log statistics
    logger.info(f"Engagement Rate - Min: {df['Engagement_Rate'].min():.4f}, "
                f"Max: {df['Engagement_Rate'].max():.4f}, "
                f"Mean: {df['Engagement_Rate'].mean():.4f}")
    
    # Sample if needed for performance
    MAX_ROWS = 100000
    original_size = len(df)
    
    if len(df) > MAX_ROWS:
        df_sampled = df.sample(n=MAX_ROWS, random_state=42).reset_index(drop=True)
        logger.info(f"Dataset disampling dari {original_size:,} ke {MAX_ROWS:,} baris")
        
        st.warning(f"""
        # ⚠️ Dataset terlalu besar ({original_size:,} rows)
        
        **Aksi:** Disampling ke {MAX_ROWS:,} rows untuk performa optimal.
        
        *Untuk analisis lengkap, pertimbangkan untuk menggunakan subset data atau meningkatkan resources.*
        """)
        
        df = df_sampled
    
    # Add categorical columns if missing (for demo purposes)
    if 'ContentType' not in df.columns:
        logger.info("Menambahkan kolom ContentType untuk demo")
        # Create ContentType based on engagement rate percentiles
        df['ContentType'] = pd.qcut(df['Engagement_Rate'], 
                                   q=3, 
                                   labels=['Low Engagement', 'Medium Engagement', 'High Engagement'])
    
    logger.info(f"Data loading selesai. Final shape: {df.shape}")
    return df

@st.cache_data
def perform_clustering(df: pd.DataFrame, n_clusters: int, features_cols: list) -> Optional[Dict[str, Any]]:
    """
    Perform K-Means clustering dengan error handling komprehensif
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset yang sudah dipreprocess
    n_clusters : int
        Jumlah cluster yang diinginkan
    features_cols : list
        List kolom feature untuk clustering
        
    Returns:
    --------
    Optional[Dict[str, Any]]: Dictionary hasil clustering atau None jika gagal
    """
    
    logger.info(f"Memulai clustering dengan K={n_clusters}, features={len(features_cols)}, n_samples={len(df)}")
    
    # ==================== VALIDASI INPUT LENGKAP ====================
    validation_errors = []
    validation_warnings = []
    
    try:
        # 1. Validasi parameter dasar
        if n_clusters < 2:
            validation_errors.append(f"n_clusters ({n_clusters}) harus >= 2")
        
        if n_clusters > 20:
            validation_errors.append(f"n_clusters ({n_clusters}) terlalu besar, maksimum 20")
        
        if len(df) < n_clusters:
            validation_errors.append(
                f"Jumlah sampel ({len(df)}) kurang dari n_clusters ({n_clusters})"
            )
        
        if len(df) < 10:
            validation_errors.append("Dataset terlalu kecil untuk clustering (minimum 10 baris)")
        
        # 2. Validasi features_cols
        missing_features = [col for col in features_cols if col not in df.columns]
        if missing_features:
            validation_errors.append(f"Feature tidak ditemukan: {missing_features}")
        
        # 3. Validasi tipe data features
        if features_cols:
            numeric_features = df[features_cols].select_dtypes(include=[np.number]).columns.tolist()
            non_numeric = [col for col in features_cols if col not in numeric_features]
            if non_numeric:
                validation_errors.append(f"Feature non-numerik: {non_numeric}")
        
        # 4. Validasi missing values
        if features_cols:
            missing_counts = df[features_cols].isnull().sum()
            total_missing = missing_counts.sum()
            if total_missing > 0:
                logger.warning(f"Ditemukan {total_missing} missing values di features")
                validation_warnings.append(f"{total_missing} missing values ditemukan")
                
                # Cek persentase missing
                missing_pct = (total_missing / (len(df) * len(features_cols))) * 100
                if missing_pct > 30:
                    validation_errors.append(f"Missing values terlalu tinggi ({missing_pct:.1f}%)")
        
        # 5. Validasi zero/low variance
        if features_cols and len(df) > 1:
            variances = df[features_cols].var()
            zero_var_features = variances[variances == 0].index.tolist()
            low_var_features = variances[variances < 1e-10].index.tolist()
            
            if zero_var_features:
                validation_errors.append(f"Feature zero variance: {zero_var_features}")
            
            if low_var_features and len(low_var_features) == len(features_cols):
                validation_errors.append("Semua features memiliki variance sangat rendah")
            elif low_var_features:
                validation_warnings.append(f"Low variance features: {low_var_features}")
        
        # 6. Validasi infinity/nan
        if features_cols:
            infinite_mask = ~np.isfinite(df[features_cols].values)
            if infinite_mask.any():
                infinite_count = infinite_mask.sum()
                validation_errors.append(f"Ditemukan {infinite_count} nilai infinite/NaN di features")
        
        # 7. Validasi skala data (range checking)
        if features_cols:
            for col in features_cols:
                if col in df.columns:
                    col_min = df[col].min()
                    col_max = df[col].max()
                    if abs(col_max - col_min) < 1e-10:
                        validation_warnings.append(f"Feature {col} memiliki range sangat kecil")
        
        # Jika ada validation errors, raise exception
        if validation_errors:
            error_msg = " | ".join(validation_errors)
            logger.error(f"Validasi gagal: {error_msg}")
            raise ValueError(f"Validasi input gagal: {error_msg}")
        
        logger.info(f"✅ Semua validasi passed. Warnings: {validation_warnings}")
        
        # ==================== PREPROCESSING DENGAN ERROR HANDLING ====================
        features = df[features_cols].copy()
        
        # Handle missing values (jika masih ada)
        if features.isnull().any().any():
            missing_before = features.isnull().sum().sum()
            features = features.fillna(features.median())
            missing_after = features.isnull().sum().sum()
            logger.info(f"Mengisi {missing_before} missing values dengan median")
        
        # Handle infinite values
        if not np.isfinite(features.values).all():
            logger.warning("Mendeteksi nilai infinite, melakukan cleaning")
            features = features.replace([np.inf, -np.inf], np.nan)
            features = features.fillna(features.median())
        
        # ==================== STANDARDIZATION DENGAN ERROR HANDLING ====================
        scaler = None
        scaled_features = None
        
        try:
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features)
            
            # Validasi hasil scaling
            if not np.isfinite(scaled_features).all():
                raise ValueError("Nilai infinite setelah scaling")
                
            if np.isnan(scaled_features).any():
                raise ValueError("NaN setelah scaling")
                
            logger.info("StandardScaler berhasil")
                
        except Exception as e:
            logger.error(f"Error pada StandardScaler: {str(e)}")
            
            # Fallback: gunakan RobustScaler
            try:
                logger.info("Menggunakan RobustScaler sebagai fallback")
                scaler = RobustScaler()
                scaled_features = scaler.fit_transform(features)
                logger.info("RobustScaler berhasil")
            except Exception as e2:
                logger.error(f"RobustScaler juga gagal: {str(e2)}")
                raise ValueError(f"Kedua scaler gagal: {str(e)} | {str(e2)}")
        
        # ==================== CLUSTERING DENGAN ERROR HANDLING ====================
        kmeans = None
        clusters = None
        
        try:
            # Pilih algoritma berdasarkan ukuran data
            if len(scaled_features) > 10000:
                logger.info("Dataset besar, menggunakan MiniBatchKMeans")
                from sklearn.cluster import MiniBatchKMeans
                kmeans = MiniBatchKMeans(
                    n_clusters=n_clusters,
                    random_state=42,
                    batch_size=1000,
                    n_init=3,
                    max_iter=300,
                    verbose=0
                )
            else:
                kmeans = KMeans(
                    n_clusters=n_clusters,
                    random_state=42,
                    n_init=10,
                    max_iter=300,
                    verbose=0,
                    tol=1e-4
                )
            
            # Fit clustering
            clusters = kmeans.fit_predict(scaled_features)
            
            # Validasi hasil clustering
            unique_clusters = np.unique(clusters)
            clusters_formed = len(unique_clusters)
            
            if clusters_formed != n_clusters:
                logger.warning(
                    f"Hanya {clusters_formed} cluster terbentuk dari {n_clusters} yang diminta"
                )
                validation_warnings.append(f"Hanya {clusters_formed} cluster terbentuk")
            
            logger.info(f"Clustering berhasil. Cluster terbentuk: {clusters_formed}")
            
        except Exception as e:
            logger.error(f"Error pada clustering: {str(e)}")
            
            # Fallback: coba dengan jumlah cluster lebih kecil
            if n_clusters > 2:
                logger.info(f"Mencoba clustering dengan K={n_clusters-1}")
                try:
                    kmeans = KMeans(
                        n_clusters=n_clusters-1,
                        random_state=42,
                        n_init=10
                    )
                    clusters = kmeans.fit_predict(scaled_features)
                    n_clusters = n_clusters - 1
                    validation_warnings.append(f"Diubah ke K={n_clusters} karena konvergensi")
                    logger.info(f"Fallback clustering berhasil dengan K={n_clusters}")
                except Exception as e2:
                    logger.error(f"Fallback juga gagal: {str(e2)}")
                    raise ValueError(f"Clustering gagal: {str(e)}")
            else:
                raise
        
        # ==================== CALCULATE METRICS DENGAN ERROR HANDLING ====================
        metrics = {}
        
        try:
            # Silhouette score dengan sampling untuk dataset besar
            if len(scaled_features) >= 2 and len(np.unique(clusters)) >= 2:
                sample_size = min(5000, len(scaled_features))
                if len(scaled_features) > sample_size:
                    indices = np.random.choice(len(scaled_features), sample_size, replace=False)
                    silhouette = silhouette_score(scaled_features[indices], clusters[indices])
                    logger.info(f"Silhouette dihitung dengan sampling {sample_size} data")
                else:
                    silhouette = silhouette_score(scaled_features, clusters)
                
                metrics['silhouette'] = silhouette
            else:
                metrics['silhouette'] = -1
                validation_warnings.append("Silhouette tidak dapat dihitung (terlalu sedikit data/cluster)")
                
        except Exception as e:
            logger.warning(f"Tidak dapat menghitung silhouette score: {str(e)}")
            metrics['silhouette'] = -1
        
        try:
            # Davies-Bouldin score
            if len(scaled_features) >= 2 and len(np.unique(clusters)) >= 2:
                davies_bouldin = davies_bouldin_score(scaled_features, clusters)
                metrics['davies_bouldin'] = davies_bouldin
            else:
                metrics['davies_bouldin'] = float('inf')
        except Exception as e:
            logger.warning(f"Tidak dapat menghitung Davies-Bouldin: {str(e)}")
            metrics['davies_bouldin'] = float('inf')
        
        try:
            # Inertia
            metrics['inertia'] = kmeans.inertia_ if kmeans else 0
        except:
            metrics['inertia'] = 0
        
        try:
            # Cluster sizes and balance
            cluster_sizes = np.bincount(clusters)
            metrics['cluster_sizes'] = cluster_sizes
            if len(cluster_sizes) > 0 and np.mean(cluster_sizes) > 0:
                metrics['cluster_balance'] = np.std(cluster_sizes) / np.mean(cluster_sizes)
            else:
                metrics['cluster_balance'] = 0
        except:
            metrics['cluster_sizes'] = np.array([])
            metrics['cluster_balance'] = 0
        
        # ==================== PCA VISUALIZATION DENGAN ERROR HANDLING ====================
        pca_result = None
        pca_explained = None
        use_sample = False
        sample_indices = None
        
        try:
            if len(scaled_features) > 5000:
                sample_idx = np.random.choice(len(scaled_features), 5000, replace=False)
                pca = PCA(n_components=2)
                pca_result_sample = pca.fit_transform(scaled_features[sample_idx])
                pca_result = np.zeros((len(scaled_features), 2))
                pca_result[sample_idx] = pca_result_sample
                pca_explained = pca.explained_variance_ratio_
                use_sample = True
                sample_indices = sample_idx
                logger.info(f"PCA dihitung dengan sampling 5000 data")
            elif len(scaled_features) >= 2:
                pca = PCA(n_components=2)
                pca_result = pca.fit_transform(scaled_features)
                pca_explained = pca.explained_variance_ratio_
                use_sample = False
                sample_indices = None
            else:
                raise ValueError("Data terlalu sedikit untuk PCA")
            
        except Exception as e:
            logger.error(f"Error pada PCA: {str(e)}")
            # Buat dummy PCA result
            pca_result = np.random.randn(len(scaled_features), 2) * 0.1
            pca_explained = [0.5, 0.3]
            use_sample = False
            validation_warnings.append("PCA menggunakan data dummy")
        
        # ==================== RETURN RESULT ====================
        result = {
            'clusters': clusters,
            'kmeans': kmeans,
            'scaler': scaler,
            'scaled_features': scaled_features,
            'pca_result': pca_result,
            'pca_explained': pca_explained,
            'use_sample': use_sample,
            'sample_indices': sample_indices,
            'metrics': metrics,
            'validation_info': {
                'n_samples': len(df),
                'n_features': len(features_cols),
                'features_used': features_cols,
                'clusters_requested': n_clusters,
                'clusters_formed': len(np.unique(clusters)),
                'warnings': validation_warnings
            },
            'success': True
        }
        
        logger.info(f"✅ Clustering selesai. Silhouette: {metrics.get('silhouette', 'N/A'):.3f}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Error fatal dalam clustering: {str(e)}", exc_info=True)
        
        # Return minimal error result untuk graceful degradation
        error_result = {
            'clusters': np.zeros(len(df), dtype=int),  # Semua cluster 0
            'kmeans': None,
            'scaler': None,
            'scaled_features': None,
            'pca_result': np.random.randn(len(df), 2) * 0.1,
            'pca_explained': [0.5, 0.3],
            'use_sample': False,
            'sample_indices': None,
            'metrics': {
                'silhouette': -1,
                'davies_bouldin': float('inf'),
                'inertia': 0,
                'cluster_sizes': np.array([len(df)]),
                'cluster_balance': 0
            },
            'error': str(e),
            'validation_errors': validation_errors,
            'validation_warnings': validation_warnings,
            'validation_info': {
                'n_samples': len(df),
                'n_features': len(features_cols),
                'features_used': features_cols,
                'clusters_requested': n_clusters,
                'status': 'ERROR',
                'error_message': str(e)
            },
            'success': False,
            'fallback': True
        }
        
        return error_result

def display_clustering_diagnostics(df: pd.DataFrame, result: Dict, features_cols: list):
    """
    Tampilkan diagnostic informasi clustering
    """
    with st.expander("🔍 Diagnostics & Warnings", expanded=False):
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Data quality
            st.markdown("#### 📊 Data Quality")
            
            # Missing values
            missing_pct = (df[features_cols].isnull().sum().sum() / 
                          (len(df) * len(features_cols))) * 100
            delta_color = "normal" if missing_pct < 5 else "off"
            st.metric("Missing Values", f"{missing_pct:.1f}%", 
                     delta="Good" if missing_pct < 5 else "Check",
                     delta_color=delta_color)
            
            # Zero variance features
            variances = df[features_cols].var()
            zero_var = variances[variances == 0].index.tolist()
            delta_color = "normal" if len(zero_var) == 0 else "off"
            st.metric("Zero Variance Features", len(zero_var),
                     delta="OK" if len(zero_var) == 0 else "Warning",
                     delta_color=delta_color)
            
            # Data size
            st.metric("Data Points", f"{len(df):,}",
                     delta="Good" if len(df) >= 100 else "Small",
                     delta_color="normal" if len(df) >= 100 else "off")
            
        with col2:
            # Clustering quality
            st.markdown("#### 🎯 Clustering Quality")
            
            # Cluster balance
            if 'cluster_sizes' in result.get('metrics', {}):
                cluster_sizes = result['metrics']['cluster_sizes']
                if len(cluster_sizes) > 0 and np.mean(cluster_sizes) > 0:
                    balance = np.std(cluster_sizes) / np.mean(cluster_sizes)
                    delta_color = "normal" if balance < 0.5 else "off"
                    st.metric("Cluster Balance", f"{balance:.2f}",
                             delta="Balanced" if balance < 0.5 else "Unbalanced",
                             delta_color=delta_color)
            
            # Silhouette score interpretation
            silhouette = result.get('metrics', {}).get('silhouette', -1)
            if silhouette >= 0:
                if silhouette > 0.5:
                    quality = "Good"
                    delta_color = "normal"
                elif silhouette > 0.25:
                    quality = "Fair" 
                    delta_color = "off"
                else:
                    quality = "Poor"
                    delta_color = "off"
                    
                st.metric("Silhouette Quality", quality,
                         delta=f"{silhouette:.3f}",
                         delta_color=delta_color)
            else:
                st.metric("Silhouette", "N/A", delta="Not calculated")
            
            # Clusters formed
            clusters_formed = len(np.unique(result.get('clusters', [])))
            clusters_requested = result.get('validation_info', {}).get('clusters_requested', 0)
            if clusters_formed == clusters_requested:
                st.metric("Clusters Formed", clusters_formed, delta="Complete")
            else:
                st.metric("Clusters Formed", clusters_formed, 
                         delta=f"Requested: {clusters_requested}",
                         delta_color="off")
        
        # Warnings section
        warnings_list = []
        
        # Check for small clusters
        if 'cluster_sizes' in result.get('metrics', {}):
            cluster_sizes = result['metrics']['cluster_sizes']
            small_clusters = (cluster_sizes < 5).sum()
            if small_clusters > 0:
                warnings_list.append(f"⚠️ {small_clusters} cluster memiliki <5 data points")
        
        # Check silhouette score
        if silhouette < 0.1 and silhouette >= 0:
            warnings_list.append(f"⚠️ Silhouette score rendah ({silhouette:.3f})")
        
        # Check if using fallback
        if result.get('fallback', False):
            warnings_list.append("⚠️ Menggunakan fallback clustering")
        
        # Check validation warnings
        if 'validation_info' in result and 'warnings' in result['validation_info']:
            for warning in result['validation_info']['warnings']:
                warnings_list.append(f"⚠️ {warning}")
        
        # Check for any errors
        if result.get('error'):
            warnings_list.append(f"❌ Error: {result['error']}")
        
        # Display warnings
        if warnings_list:
            st.markdown("---")
            st.markdown("#### ⚠️ Warnings & Issues")
            for warning in warnings_list[:5]:  # Show max 5 warnings
                st.warning(warning)
            
            if len(warnings_list) > 5:
                st.info(f"... dan {len(warnings_list) - 5} warnings lainnya")
        else:
            st.success("✅ Tidak ada warning yang signifikan")
        
        # Tips for improvement
        if silhouette < 0.3 or result.get('fallback', False):
            st.markdown("---")
            st.markdown("#### 💡 Tips untuk Perbaikan")
            
            tips = [
                "Coba kurangi jumlah cluster (K)",
                "Periksa outliers di data Anda",
                "Pastikan semua features memiliki variasi yang cukup",
                "Coba features yang berbeda untuk clustering",
                "Jika data besar, coba sampling untuk testing"
            ]
            
            for tip in tips:
                st.markdown(f"• {tip}")

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
        
        # Get suggestions for optimal K
        df_loaded = False
        try:
            df = load_data()
            df_loaded = True
            features_cols = ['Likes', 'Shares', 'Comments', 'Views', 'TimeSpentOnContent', 'Engagement_Rate']
            suggestions = suggest_optimal_clusters(df, features_cols)
        except:
            df_loaded = False
            suggestions = {'recommended': 4, 'recommended_range': '2-8'}
        
        # K value slider with suggestions
        default_k = suggestions.get('recommended', 4)
        
        k_value = st.slider(
            "Jumlah Cluster (K):",
            min_value=2,
            max_value=10,
            value=default_k,
            help=f"Disarankan: {suggestions.get('recommended_range', '2-8')}. K={default_k} berdasarkan data"
        )
        
        # Show suggestions
        if suggestions.get('warnings'):
            with st.expander("📋 Saran & Peringatan"):
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
                    <p class='info-value {'success' if missing_values == 0 else 'warning' if missing_values < 10 else 'error'}'>
                        {missing_values}
                    </p>
                </div>
                <div class='info-item'>
                    <p class='info-label'>Engagement Rate</p>
                    <p class='info-value success'>✓ Aktif</p>
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
                with st.expander("ℹ️ Validation Warnings", expanded=False):
                    for warning in validation_warnings[:3]:
                        st.warning(warning)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ==================== CLUSTERING DENGAN ERROR HANDLING ====================
    with st.spinner("🔄 Melakukan clustering..."):
        progress_bar = st.progress(0)
        
        try:
            # Update progress
            progress_bar.progress(20)
            
            # Validasi data sebelum clustering
            is_valid, validation_msg, validation_warnings = validate_data_for_clustering(df, features_cols)
            
            if not is_valid:
                progress_bar.progress(100)
                st.error(f"❌ {validation_msg}")
                
                with st.expander("💡 Tips perbaikan data"):
                    st.markdown("""
                    1. **Cek missing values**: Pastikan tidak ada kolom dengan >50% missing
                    2. **Cek tipe data**: Semua feature harus numerik
                    3. **Cek variance**: Pastikan features memiliki variasi data
                    4. **Cek outliers**: Outlier ekstrem dapat mempengaruhi clustering
                    5. **Minimum data**: Minimal 10 data points untuk clustering
                    """)
                
                # Suggest optimal clusters
                suggestions = suggest_optimal_clusters(df, features_cols)
                st.info(f"**Saran:** Untuk {len(df):,} data, coba K={suggestions.get('recommended', 4)}")
                
                st.stop()
            
            progress_bar.progress(40)
            
            # Tampilkan info validasi
            if validation_warnings:
                st.warning(f"⚠️ {len(validation_warnings)} warnings ditemukan. Clustering tetap dilanjutkan.")
            
            progress_bar.progress(60)
            
            # Lakukan clustering
            result = perform_clustering(df, k_value, features_cols)
            
            progress_bar.progress(80)
            
            # Cek jika clustering menghasilkan error
            if result and not result.get('success', True):
                st.error(f"❌ Error dalam clustering: {result.get('error', 'Unknown error')}")
                
                # Fallback: coba dengan K yang lebih kecil
                if k_value > 2:
                    st.warning(f"⏳ Mencoba clustering dengan K={k_value-1}...")
                    result = perform_clustering(df, k_value-1, features_cols)
                    
                    if not result.get('success', True):
                        st.error("❌ Clustering tetap gagal. Silakan cek data Anda.")
                        st.stop()
                    else:
                        st.success(f"✅ Clustering berhasil dengan K={k_value-1}")
                        k_value = k_value - 1  # Update K value
                else:
                    st.error("❌ Tidak dapat melakukan clustering. Cek data dan coba lagi.")
                    st.stop()
            
            progress_bar.progress(100)
            
        except Exception as e:
            progress_bar.progress(100)
            logger.error(f"Exception tidak terduga: {str(e)}", exc_info=True)
            st.error(f"❌ Exception tidak terduga: {str(e)}")
            
            # Fallback: buat cluster dummy untuk menjaga aplikasi tetap berjalan
            st.warning("🔄 Membuat cluster dummy untuk melanjutkan...")
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
    display_clustering_diagnostics(df, result, features_cols)
    
    # ==================== METRICS DENGAN ERROR INDICATOR ====================
    st.markdown("""
    <div class='custom-card'>
        <h3 style='color: #1E293B; margin: 0 0 1.5rem 0;'>
            📈 Model Performance Metrics
        </h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Tampilkan indicator jika ada error atau fallback
    if result.get('fallback', False):
        st.warning("⚠️ Menggunakan hasil fallback clustering. Metrics mungkin tidak akurat.")
    
    if result.get('error'):
        st.error(f"⚠️ Clustering memiliki masalah: {result.get('error', 'Unknown error')}")
    
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
            Built with Streamlit & Scikit-learn | Enhanced Error Handling v2.1
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
        ❌ Critical Error: {str(e)}
        
        Aplikasi mengalami error yang tidak dapat dipulihkan.
        
        **Langkah troubleshooting:**
        1. Cek file log: tiktok_dashboard.log
        2. Pastikan dataset ada dan formatnya benar
        3. Restart aplikasi
        4. Hubungi developer jika error berlanjut
        """)