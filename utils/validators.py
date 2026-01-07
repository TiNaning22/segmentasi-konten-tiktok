import pandas as pd
import numpy as np
from typing import Tuple, List, Dict

def validate_data_for_clustering(df: pd.DataFrame, features_cols: list) -> Tuple[bool, str, List[str]]:
    
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
    
    # 4. Cek tipe data numerik dengan cara yang aman
    non_numeric = []
    for col in features_cols:
        if col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                non_numeric.append(col)
    
    if non_numeric:
        return False, f"❌ Kolom non-numerik: {non_numeric}", []
    
    # 5. Cek missing values percentage
    # Gunakan hanya features yang ada
    existing_features = [col for col in features_cols if col in df.columns]
    if existing_features:
        missing_pct = (df[existing_features].isna().sum().sum() / 
                      (len(df) * len(existing_features))) * 100
        if missing_pct > 50:
            return False, f"❌ Missing values terlalu tinggi ({missing_pct:.1f}%)", []
        elif missing_pct > 10:
            warnings.append(f"Missing values: {missing_pct:.1f}%")
    
    # 6. Cek zero variance dengan cara yang aman
    if existing_features and len(df) > 1:
        variances = df[existing_features].var(ddof=0)
        # Handle kasus variances adalah Series atau scalar
        if isinstance(variances, pd.Series):
            zero_var_features = variances[variances == 0].index.tolist()
        else:
            # Jika hanya satu feature, variances adalah scalar
            zero_var_features = existing_features if variances == 0 else []
        
        if len(zero_var_features) == len(existing_features):
            return False, "❌ Semua features memiliki zero variance", []
        elif zero_var_features:
            warnings.append(f"Zero variance features: {zero_var_features}")
    
    # 7. Cek outliers ekstrem (optional warning)
    extreme_outlier_cols = []
    for col in existing_features:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            # Pastikan kita punya cukup data
            if len(df[col].dropna()) > 1:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                if iqr > 0:  # Hindari division by zero
                    lower_bound = q1 - 10 * iqr
                    upper_bound = q3 + 10 * iqr
                    
                    # Gunakan .any() untuk array boolean
                    extreme_outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).any()
                    if extreme_outliers:
                        extreme_outlier_cols.append(f"{col}: outliers detected")
    
    if extreme_outlier_cols:
        warnings.append(f"Extreme outliers detected in: {', '.join(extreme_outlier_cols[:3])}")
    
    # 8. Cek korelasi sangat tinggi antar features
    if len(existing_features) > 1:
        # Gunakan hanya data numerik
        numeric_df = df[existing_features].select_dtypes(include=[np.number])
        if len(numeric_df.columns) > 1:
            corr_matrix = numeric_df.corr().abs()
            np.fill_diagonal(corr_matrix.values, 0)
            
            high_corr_pairs = []
            features = corr_matrix.columns.tolist()
            
            for i in range(len(features)):
                for j in range(i+1, len(features)):
                    corr_value = corr_matrix.iloc[i, j]
                    if corr_value > 0.95:
                        high_corr_pairs.append(f"{features[i]}-{features[j]}: {corr_value:.2f}")
            
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