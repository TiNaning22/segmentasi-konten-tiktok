import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
import logging

logger = logging.getLogger(__name__)

def perform_clustering(df: pd.DataFrame, n_clusters: int, features_cols: list) -> Optional[Dict[str, Any]]:
    """
    Perform K-Means clustering dengan error handling komprehensif
    """
    logger.info(f"Memulai clustering dengan K={n_clusters}, features={len(features_cols)}, n_samples={len(df)}")
    
    validation_errors = []
    validation_warnings = []
    
    try:
        # ==================== VALIDASI INPUT ====================
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
        
        # Cek features
        missing_features = [col for col in features_cols if col not in df.columns]
        if missing_features:
            validation_errors.append(f"Feature tidak ditemukan: {missing_features}")
        
        # Cek tipe data
        if features_cols:
            numeric_features = df[features_cols].select_dtypes(include=[np.number]).columns.tolist()
            non_numeric = [col for col in features_cols if col not in numeric_features]
            if non_numeric:
                validation_errors.append(f"Feature non-numerik: {non_numeric}")
        
        # Cek missing values
        if features_cols:
            missing_counts = df[features_cols].isnull().sum()
            total_missing = missing_counts.sum()
            if total_missing > 0:
                missing_pct = (total_missing / (len(df) * len(features_cols))) * 100
                if missing_pct > 30:
                    validation_errors.append(f"Missing values terlalu tinggi ({missing_pct:.1f}%)")
        
        # Cek zero variance
        if features_cols and len(df) > 1:
            variances = df[features_cols].var()
            zero_var_features = variances[variances == 0].index.tolist()
            if zero_var_features:
                validation_errors.append(f"Feature zero variance: {zero_var_features}")
        
        if validation_errors:
            error_msg = " | ".join(validation_errors)
            raise ValueError(f"Validasi input gagal: {error_msg}")
        
        logger.info(f"Semua validasi passed. Warnings: {validation_warnings}")
        
        # ==================== PREPROCESSING ====================
        features = df[features_cols].copy()
        
        # Handle missing values
        if features.isnull().any().any():
            features = features.fillna(features.median())
        
        # Handle infinite values
        if not np.isfinite(features.values).all():
            features = features.replace([np.inf, -np.inf], np.nan)
            features = features.fillna(features.median())
        
        # ==================== STANDARDIZATION ====================
        scaler = None
        scaled_features = None
        
        try:
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features)
        except:
            try:
                scaler = RobustScaler()
                scaled_features = scaler.fit_transform(features)
            except Exception as e:
                raise ValueError(f"Kedua scaler gagal: {str(e)}")
        
        # ==================== CLUSTERING ====================
        if len(scaled_features) > 10000:
            kmeans = MiniBatchKMeans(
                n_clusters=n_clusters,
                random_state=42,
                batch_size=1000,
                n_init=3,
                max_iter=300
            )
        else:
            kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=42,
                n_init=10,
                max_iter=300
            )
        
        clusters = kmeans.fit_predict(scaled_features)
        unique_clusters = np.unique(clusters)
        
        if len(unique_clusters) != n_clusters:
            validation_warnings.append(f"Hanya {len(unique_clusters)} cluster terbentuk")
        
        # ==================== METRICS ====================
        metrics = {}
        
        try:
            if len(scaled_features) >= 2 and len(unique_clusters) >= 2:
                sample_size = min(5000, len(scaled_features))
                if len(scaled_features) > sample_size:
                    indices = np.random.choice(len(scaled_features), sample_size, replace=False)
                    silhouette = silhouette_score(scaled_features[indices], clusters[indices])
                else:
                    silhouette = silhouette_score(scaled_features, clusters)
                metrics['silhouette'] = silhouette
            else:
                metrics['silhouette'] = -1
        except:
            metrics['silhouette'] = -1
        
        try:
            if len(scaled_features) >= 2 and len(unique_clusters) >= 2:
                metrics['davies_bouldin'] = davies_bouldin_score(scaled_features, clusters)
            else:
                metrics['davies_bouldin'] = float('inf')
        except:
            metrics['davies_bouldin'] = float('inf')
        
        metrics['inertia'] = kmeans.inertia_ if hasattr(kmeans, 'inertia_') else 0
        
        try:
            cluster_sizes = np.bincount(clusters)
            metrics['cluster_sizes'] = cluster_sizes
            if len(cluster_sizes) > 0 and np.mean(cluster_sizes) > 0:
                metrics['cluster_balance'] = np.std(cluster_sizes) / np.mean(cluster_sizes)
            else:
                metrics['cluster_balance'] = 0
        except:
            metrics['cluster_sizes'] = np.array([])
            metrics['cluster_balance'] = 0
        
        # ==================== PCA VISUALIZATION ====================
        pca_result = None
        pca_explained = None
        
        try:
            if len(scaled_features) > 5000:
                sample_idx = np.random.choice(len(scaled_features), 5000, replace=False)
                pca = PCA(n_components=2)
                pca_result_sample = pca.fit_transform(scaled_features[sample_idx])
                pca_result = np.zeros((len(scaled_features), 2))
                pca_result[sample_idx] = pca_result_sample
                pca_explained = pca.explained_variance_ratio_
            elif len(scaled_features) >= 2:
                pca = PCA(n_components=2)
                pca_result = pca.fit_transform(scaled_features)
                pca_explained = pca.explained_variance_ratio_
            else:
                raise ValueError("Data terlalu sedikit untuk PCA")
        except:
            pca_result = np.random.randn(len(scaled_features), 2) * 0.1
            pca_explained = [0.5, 0.3]
            validation_warnings.append("PCA menggunakan data dummy")
        
        # ==================== RETURN RESULT ====================
        result = {
            'clusters': clusters,
            'kmeans': kmeans,
            'scaler': scaler,
            'scaled_features': scaled_features,
            'pca_result': pca_result,
            'pca_explained': pca_explained,
            'metrics': metrics,
            'validation_info': {
                'n_samples': len(df),
                'n_features': len(features_cols),
                'features_used': features_cols,
                'clusters_requested': n_clusters,
                'clusters_formed': len(unique_clusters),
                'warnings': validation_warnings
            },
            'use_sample': len(df) > 10000,
            'sample_indices': None,
            'success': True
        }
        
        logger.info(f"Clustering selesai. Silhouette: {metrics.get('silhouette', 'N/A'):.3f}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error fatal dalam clustering: {str(e)}", exc_info=True)
        
        error_result = {
            'clusters': np.zeros(len(df), dtype=int),
            'kmeans': None,
            'scaler': None,
            'scaled_features': None,
            'pca_result': np.random.randn(len(df), 2) * 0.1,
            'pca_explained': [0.5, 0.3],
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