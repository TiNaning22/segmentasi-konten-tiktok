import pandas as pd
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

def retrain_model(new_data_path=None):
    """
    Retrain model dengan data baru (jika ada)
    """
    # Load data lama
    base_df = pd.read_csv('tiktok_digital_marketing_data.csv')
    
    # Jika ada data baru, gabungkan
    if new_data_path and os.path.exists(new_data_path):
        new_df = pd.read_csv(new_data_path)
        # Pastikan kolom sama
        if set(new_df.columns) == set(base_df.columns):
            combined_df = pd.concat([base_df, new_df], ignore_index=True)
            print(f"✅ Data baru ditambahkan. Total data: {len(combined_df)}")
        else:
            combined_df = base_df
            print("⚠️ Struktur data baru tidak cocok, menggunakan data lama saja.")
    else:
        combined_df = base_df
    
    # Fitur untuk clustering
    features = combined_df[['Likes', 'Shares', 'Comments', 'Views', 'TimeSpentOnContent']]
    
    # Standardisasi
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Tentukan k optimal (bisa diotomatisasi)
    from sklearn.metrics import silhouette_score
    
    best_k = 4
    best_score = -1
    
    for k in range(2, 7):
        kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans_temp.fit_predict(scaled_features)
        score = silhouette_score(scaled_features, labels)
        
        if score > best_score:
            best_score = score
            best_k = k
    
    print(f"🔍 K optimal: {best_k} (Silhouette Score: {best_score:.3f})")
    
    # Training final model
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    kmeans.fit(scaled_features)
    
    # Save model dan scaler
    joblib.dump(kmeans, 'kmeans_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    
    # Update data dengan cluster baru
    combined_df['Cluster'] = kmeans.predict(scaled_features)
    combined_df.to_csv('tiktok_with_clusters_updated.csv', index=False)
    
    print(f"✅ Model retrained dengan {best_k} cluster!")
    return best_k