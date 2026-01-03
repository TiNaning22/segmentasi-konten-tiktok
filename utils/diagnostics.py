import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List

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
            
            # Silhouette score
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
        
        if result.get('error'):
            warnings_list.append(f"❌ Error: {result['error']}")
        
        # Display warnings
        if warnings_list:
            st.markdown("---")
            st.markdown("#### ⚠️ Warnings & Issues")
            for warning in warnings_list[:5]:
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