import streamlit as st

def load_css():
    """Load custom CSS styles"""
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