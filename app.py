import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Konfigurasi halaman
st.set_page_config(
    page_title="TikTok Content Segmenter",
    page_icon="📊",
    layout="wide"
)

# Judul dan deskripsi
st.title("🎯 TikTok Content Segmentation - K-Means Clustering")
st.markdown("""
Segmentasikan konten TikTok berdasarkan performa metrik. 
**Upload data Anda** atau **input manual** untuk mendapatkan segmentasi real-time.
""")

# Sidebar untuk navigasi
st.sidebar.header("⚙️ Mode Input Data")

# Mode input: Upload atau Manual
input_mode = st.sidebar.radio(
    "Pilih mode input:",
    ["📤 Upload CSV File", "✍️ Input Manual"]
)

# Load model dan scaler yang sudah ditraining
@st.cache_resource
def load_model():
    kmeans = joblib.load('kmeans_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return kmeans, scaler

kmeans, scaler = load_model()

# Fungsi untuk memproses data baru
def predict_cluster(data):
    """Prediksi cluster untuk data baru"""
    # Scaler transform
    scaled_data = scaler.transform(data)
    # Predict cluster
    clusters = kmeans.predict(scaled_data)
    return clusters

# ================= MODE UPLOAD CSV =================
if input_mode == "📤 Upload CSV File":
    st.header("📤 Upload Data TikTok Anda")
    
    uploaded_file = st.file_uploader(
        "Pilih file CSV dengan kolom: Likes, Shares, Comments, Views, TimeSpentOnContent",
        type=['csv']
    )
    
    if uploaded_file is not None:
        try:
            # Baca file
            user_df = pd.read_csv(uploaded_file)
            st.success(f"✅ File berhasil diunggah! {len(user_df)} baris data ditemukan.")
            
            # Tampilkan preview
            with st.expander("👁️ Preview Data"):
                st.dataframe(user_df.head())
            
            # Cek kolom yang diperlukan
            required_cols = ['Likes', 'Shares', 'Comments', 'Views', 'TimeSpentOnContent']
            missing_cols = [col for col in required_cols if col not in user_df.columns]
            
            if missing_cols:
                st.error(f"❌ Kolom berikut tidak ditemukan: {missing_cols}")
                st.info("Pastikan file Anda memiliki kolom dengan nama yang tepat.")
            else:
                # Ekstrak fitur
                features_df = user_df[required_cols]
                
                # Prediksi cluster
                if st.button("🚀 Analisis & Segmentasi", type="primary"):
                    with st.spinner("Menganalisis data..."):
                        # Prediksi
                        clusters = predict_cluster(features_df)
                        user_df['Cluster'] = clusters
                        
                        # Simpan hasil ke session state
                        st.session_state['result_df'] = user_df
                        
                    st.success(f"✅ Segmentasi selesai! {len(np.unique(clusters))} cluster ditemukan.")
                    
                    # Tampilkan hasil
                    st.subheader("📊 Hasil Segmentasi")
                    
                    # 1. Distribusi cluster
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig1 = px.pie(
                            user_df, 
                            names='Cluster', 
                            title='Distribusi Konten per Cluster',
                            color_discrete_sequence=px.colors.qualitative.Set3
                        )
                        st.plotly_chart(fig1, use_container_width=True)
                    
                    with col2:
                        cluster_stats = user_df.groupby('Cluster')[required_cols].mean()
                        fig2 = px.bar(
                            cluster_stats.T,
                            barmode='group',
                            title='Rata-rata Metrik per Cluster',
                            labels={'value': 'Rata-rata', 'variable': 'Cluster'}
                        )
                        st.plotly_chart(fig2, use_container_width=True)
                    
                    # 2. Visualisasi PCA
                    st.subheader("🎨 Visualisasi Cluster (PCA)")
                    
                    # Scaler transform untuk PCA
                    scaled_features = scaler.transform(features_df)
                    
                    # PCA untuk visualisasi 2D
                    pca = PCA(n_components=2)
                    pca_result = pca.fit_transform(scaled_features)
                    
                    pca_df = pd.DataFrame({
                        'PCA1': pca_result[:, 0],
                        'PCA2': pca_result[:, 1],
                        'Cluster': clusters
                    })
                    
                    fig3 = px.scatter(
                        pca_df, 
                        x='PCA1', 
                        y='PCA2', 
                        color='Cluster',
                        hover_data={'PCA1': ':.2f', 'PCA2': ':.2f'},
                        title='Visualisasi 2D Cluster (PCA)',
                        color_discrete_sequence=px.colors.qualitative.Set3
                    )
                    st.plotly_chart(fig3, use_container_width=True)
                    
                    # 3. Tabel detail hasil
                    st.subheader("📋 Detail Data dengan Segmentasi")
                    
                    # Filter cluster
                    selected_clusters = st.multiselect(
                        "Filter Cluster:",
                        options=sorted(user_df['Cluster'].unique()),
                        default=sorted(user_df['Cluster'].unique())
                    )
                    
                    filtered_df = user_df[user_df['Cluster'].isin(selected_clusters)]
                    st.dataframe(
                        filtered_df.style.background_gradient(
                            subset=required_cols, 
                            cmap='YlOrRd'
                        ),
                        use_container_width=True
                    )
                    
                    # 4. Download hasil
                    csv = user_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Hasil Segmentasi (CSV)",
                        data=csv,
                        file_name="tiktok_segmented.csv",
                        mime="text/csv",
                        type="primary"
                    )
                    
                    # 5. Profil cluster
                    st.subheader("🎯 Profil Rekomendasi per Cluster")
                    
                    # Analisis sederhana
                    cluster_descriptions = {
                        0: "📉 **Cluster 0 - Konten Performa Rendah**: Tingkat engagement rendah, butuh optimasi.",
                        1: "📈 **Cluster 1 - Konten Viral**: Likes dan shares tinggi, pertahankan format!",
                        2: "👁️ **Cluster 2 - Konten Informative**: Views tinggi, engagement sedang.",
                        3: "🎯 **Cluster 3 - Konten Niche**: Audience spesifik, engagement konsisten."
                    }
                    
                    for cluster_num in sorted(user_df['Cluster'].unique()):
                        with st.expander(f"📌 Cluster {cluster_num}"):
                            st.markdown(cluster_descriptions.get(cluster_num, "Cluster belum dianalisis."))
                            
                            # Stats per cluster
                            cluster_data = user_df[user_df['Cluster'] == cluster_num]
                            st.metric("Jumlah Konten", len(cluster_data))
                            
                            avg_stats = cluster_data[required_cols].mean()
                            cols = st.columns(len(required_cols))
                            for idx, col_name in enumerate(required_cols):
                                cols[idx].metric(
                                    f"Avg {col_name}",
                                    f"{avg_stats[col_name]:.0f}"
                                )
        
        except Exception as e:
            st.error(f"❌ Error membaca file: {e}")

# ================= MODE INPUT MANUAL =================
else:
    st.header("✍️ Input Data Manual")
    
    with st.form("manual_input_form"):
        st.subheader("Masukkan Metrik Konten TikTok")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            likes = st.number_input("Likes", min_value=0, value=100, step=10)
            shares = st.number_input("Shares", min_value=0, value=50, step=10)
        
        with col2:
            comments = st.number_input("Comments", min_value=0, value=20, step=5)
            views = st.number_input("Views", min_value=0, value=1000, step=100)
        
        with col3:
            time_spent = st.number_input("Time Spent (detik)", min_value=0, value=30, step=5)
        
        # Input multiple entries
        st.subheader("Input Multiple Konten (Opsional)")
        num_entries = st.slider("Jumlah konten untuk dianalisis", 1, 10, 3)
        
        manual_data = []
        for i in range(num_entries):
            st.markdown(f"**Konten {i+1}**")
            cols = st.columns(5)
            with cols[0]:
                l = st.number_input(f"Likes {i+1}", min_value=0, value=likes, key=f"likes_{i}")
            with cols[1]:
                s = st.number_input(f"Shares {i+1}", min_value=0, value=shares, key=f"shares_{i}")
            with cols[2]:
                c = st.number_input(f"Comments {i+1}", min_value=0, value=comments, key=f"comments_{i}")
            with cols[3]:
                v = st.number_input(f"Views {i+1}", min_value=0, value=views, key=f"views_{i}")
            with cols[4]:
                t = st.number_input(f"Time {i+1}", min_value=0, value=time_spent, key=f"time_{i}")
            manual_data.append([l, s, c, v, t])
        
        submitted = st.form_submit_button("🔍 Prediksi Segmentasi")
        
        if submitted:
            # Konversi ke DataFrame
            input_df = pd.DataFrame(
                manual_data,
                columns=['Likes', 'Shares', 'Comments', 'Views', 'TimeSpentOnContent']
            )
            
            # Prediksi
            clusters = predict_cluster(input_df)
            input_df['Cluster'] = clusters
            
            st.session_state['manual_result'] = input_df
            
            # Tampilkan hasil
            st.success(f"✅ Prediksi selesai! Konten dikelompokkan ke dalam cluster: {clusters}")
            
            # Tabel hasil
            st.dataframe(
                input_df.style.apply(
                    lambda x: ['background: lightgreen' if x.name == i else '' 
                              for i in range(len(x))],
                    axis=1
                )
            )
            
            # Analisis per cluster
            st.subheader("📈 Analisis Cluster")
            
            for cluster_num in np.unique(clusters):
                cluster_items = input_df[input_df['Cluster'] == cluster_num]
                st.write(f"**Cluster {cluster_num}**: {len(cluster_items)} konten")
                
                if len(cluster_items) > 0:
                    avg_values = cluster_items.mean(numeric_only=True)
                    
                    # Tampilkan metrik
                    cols = st.columns(5)
                    metrics = ['Likes', 'Shares', 'Comments', 'Views', 'TimeSpentOnContent']
                    for idx, metric in enumerate(metrics):
                        cols[idx].metric(
                            f"Avg {metric}",
                            f"{avg_values[metric]:.0f}"
                        )
                
                st.markdown("---")

# ================= BAGIAN UMUM =================
st.sidebar.markdown("---")
st.sidebar.subheader("ℹ️ Tentang Model")
st.sidebar.info("""
**Model K-Means** dengan 4 cluster:
1. Cluster 0: Low Performance
2. Cluster 1: Viral Content
3. Cluster 2: Informative
4. Cluster 3: Niche Audience

Model dilatih dengan 98 sampel data TikTok.
""")

st.sidebar.subheader("🔄 Retrain Model")
if st.sidebar.button("Retrain dengan Data Baru"):
    with st.spinner("Melatih ulang model..."):
        # Load semua data (lama + baru jika ada)
        try:
            from train_kmeans import retrain_model
            retrain_model()
            st.sidebar.success("✅ Model berhasil diperbarui!")
        except:
            st.sidebar.warning("Fitur retrain dalam pengembangan")

# Footer
st.markdown("---")
st.caption("© 2024 TikTok Content Segmentation Dashboard | Dibuat dengan Streamlit & Scikit-learn")