import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load dataset (training awal)
df = pd.read_csv('tiktok_digital_marketing_data.csv')

# Fitur untuk clustering
features = df[['Likes', 'Shares', 'Comments', 'Views', 'TimeSpentOnContent']]

# 1. Standardisasi
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# 2. Training K-Means (gunakan elbow method sebelumnya)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans.fit(scaled_features)

# 3. Simpan model dan scaler
joblib.dump(kmeans, 'kmeans_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("✅ Model dan scaler berhasil disimpan!")

# 4. Tambahkan cluster ke data awal
df['Cluster'] = kmeans.predict(scaled_features)

# Simpan data dengan cluster
df.to_csv('tiktok_with_clusters.csv', index=False)
print("✅ Data dengan cluster berhasil disimpan!")