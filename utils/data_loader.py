import pandas as pd
import numpy as np
import streamlit as st
import logging

logger = logging.getLogger(__name__)

@st.cache_data
def load_data():
    """Load dataset TikTok dengan preprocessing lengkap"""
    logger.info("Memulai loading data...")
    
    try:
        # Coba load beberapa kemungkinan file
        possible_files = [
            'tiktok_digital_marketing_data.csv',
            'tiktok_demo_data.csv',
            'tiktok_data.csv',
            'data.csv'
        ]
        
        df = None
        for file_path in possible_files:
            try:
                df = pd.read_csv(file_path)
                logger.info(f"Data berhasil dimuat dari {file_path}. Shape: {df.shape}")
                break
            except FileNotFoundError:
                continue
            except Exception as e:
                logger.warning(f"Error membaca {file_path}: {str(e)}")
                continue
        
        if df is None:
            logger.error("Tidak ada file dataset yang ditemukan")
            st.error("""
            Dataset tidak ditemukan. 
            
            **File yang dicoba:**
            - tiktok_digital_marketing_data.csv
            - tiktok_demo_data.csv  
            - tiktok_data.csv
            - data.csv
            
            Pastikan salah satu file tersebut ada di direktori yang sama.
            """)
            
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
                
                demo_data['Engagement_Rate'] = (
                    (demo_data['Likes'] + demo_data['Comments'] + demo_data['Shares']) / 
                    demo_data['Views'].clip(lower=1)
                )
                
                demo_data.to_csv('tiktok_demo_data.csv', index=False)
                st.success("Data demo berhasil dibuat. Silakan refresh halaman.")
                st.stop()
            
            st.stop()
    
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        st.error(f"Error membaca file CSV: {str(e)}")
        st.stop()
    
    # Pastikan dataframe tidak kosong
    if df is None or len(df) == 0:
        logger.error("DataFrame kosong")
        st.error("Dataset kosong atau tidak valid.")
        st.stop()
    
    # Validasi required columns
    required_cols = ['Likes', 'Shares', 'Comments', 'Views', 'TimeSpentOnContent']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        logger.warning(f"Missing required columns: {missing_cols}")
        st.warning(f"""
        Beberapa kolom yang dibutuhkan tidak ditemukan: {', '.join(missing_cols)}
        
        Aplikasi akan mencoba menggunakan kolom yang tersedia.
        """)
    
    # Handle missing values dengan aman
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) > 0:
        missing_count = df[numeric_cols].isna().sum().sum()
        if missing_count > 0:
            logger.info(f"Mengisi {missing_count} missing values dengan median")
            
            for col in numeric_cols:
                if df[col].isna().any():
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
            
            st.info(f"ℹ Mengisi {missing_count} nilai yang hilang dengan median")
    
    # Calculate engagement rate jika belum ada
    if 'Engagement_Rate' not in df.columns:
        # Cek apakah kolom yang dibutuhkan ada
        required_for_er = ['Likes', 'Comments', 'Shares', 'Views']
        if all(col in df.columns for col in required_for_er):
            df['Engagement_Rate'] = (
                (df['Likes'] + df['Comments'] + df['Shares']) / 
                df['Views'].clip(lower=1)
            )
    
    # Log statistics
    logger.info(f"Data loading selesai. Final shape: {df.shape}")
    
    return df