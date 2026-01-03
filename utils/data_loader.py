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
    MAX_ROWS = 200000
    original_size = len(df)
    
    if len(df) > MAX_ROWS:
        df_sampled = df.sample(n=MAX_ROWS, random_state=42).reset_index(drop=True)
        logger.info(f"Dataset disampling dari {original_size:,} ke {MAX_ROWS:,} baris")
        
        st.warning(f"""
        ⚠️ Dataset terlalu besar ({original_size:,} rows)
        
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