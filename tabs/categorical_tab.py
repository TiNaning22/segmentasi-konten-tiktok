import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def render(df_clustered, result, k_value, features_cols):
    """Render Profiling Kategorikal tab"""
    
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown("#### 👥 Profiling Kategorikal per Cluster")
    
    st.info("""
    **Analisis ini sesuai dengan Langkah 5:** Kolom kategorikal TIDAK dimasukkan ke dalam K-Means clustering, 
    tetapi digunakan untuk profiling dan interpretasi setelah clustering selesai. 
    Ini membantu memahami karakteristik demografis dan konten dari setiap cluster.
    """)
    
    # Deteksi kolom kategorikal secara otomatis
    categorical_cols = df_clustered.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Hapus kolom yang tidak relevan
    exclude_cols = ['Cluster']
    categorical_cols = [col for col in categorical_cols if col not in exclude_cols]
    
    if len(categorical_cols) == 0:
        st.warning("⚠️ Tidak ada kolom kategorikal yang terdeteksi di dataset.")
        st.markdown("""
        **Tips:** Dataset Anda mungkin hanya berisi data numerik. Untuk profiling kategorikal, 
        pastikan dataset memiliki kolom seperti:
        - `ContentType` (Video, Image, Text)
        - `Demographics` (Age Group, Gender, Location)
        - `Hashtag` atau `Category`
        - `DeviceType` (Mobile, Desktop)
        """)
        
        # Buat contoh data dummy untuk demo
        st.markdown("---")
        st.markdown("#### 🎭 Contoh Profiling Kategorikal (Data Demo)")
        
        # Buat data demo
        demo_data = pd.DataFrame({
            'Cluster': df_clustered['Cluster'],
            'ContentType_Demo': pd.cut(df_clustered['Likes'], bins=3, labels=['Low', 'Medium', 'High']),
            'Demographics_Demo': pd.cut(df_clustered['Views'], bins=3, labels=['Young', 'Adult', 'Senior']),
            'Platform_Demo': pd.cut(df_clustered['Engagement_Rate'], bins=3, labels=['Android', 'iOS', 'Web'])
        })
        
        categorical_cols = ['ContentType_Demo', 'Demographics_Demo', 'Platform_Demo']
        df_demo = df_clustered.copy()
        for col in categorical_cols:
            df_demo[col] = demo_data[col]
        
        analyze_categorical(df_demo, categorical_cols, k_value, is_demo=True)
        
    else:
        st.success(f"✅ Ditemukan {len(categorical_cols)} kolom kategorikal: {', '.join(categorical_cols)}")
        analyze_categorical(df_clustered, categorical_cols, k_value)
    
    st.markdown('</div>', unsafe_allow_html=True)

def analyze_categorical(df, categorical_cols, k_value, is_demo=False):
    """Analisis kategorikal untuk dataset"""
    
    # Pilih kolom kategorikal
    if is_demo:
        st.markdown("##### 📋 Analisis Profiling Kategorikal (Data Demo)")
    else:
        st.markdown("##### 📋 Analisis Profiling Kategorikal")
    
    selected_cat_col = st.selectbox(
        "Pilih Kolom Kategorikal untuk Analisis:",
        categorical_cols,
        help="Pilih kolom untuk melihat distribusinya di setiap cluster"
    )
    
    if selected_cat_col:
        # 1. Cross-Tabulation
        st.markdown(f"#### 📊 Distribusi **{selected_cat_col}** per Cluster")
        
        # Hitung crosstab (persentase)
        crosstab_pct = pd.crosstab(
            df['Cluster'], 
            df[selected_cat_col], 
            normalize='index'
        ) * 100
        
        # Hitung crosstab (jumlah)
        crosstab_count = pd.crosstab(
            df['Cluster'], 
            df[selected_cat_col]
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### 📈 Visualisasi Persentase")
            
            fig_bar = px.bar(
                crosstab_pct,
                barmode='group',
                title=f"Persentase {selected_cat_col} per Cluster (%)",
                labels={'value': 'Persentase (%)', 'variable': selected_cat_col},
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_bar.update_layout(
                height=400,
                xaxis_title="Cluster",
                yaxis_title="Persentase (%)",
                legend_title=selected_cat_col
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with col2:
            st.markdown("##### 🎯 Heatmap Distribusi")
            
            fig_heat = px.imshow(
                crosstab_pct,
                labels=dict(x=selected_cat_col, y="Cluster", color="Persentase (%)"),
                color_continuous_scale="Blues",
                aspect="auto",
                title=f"Heatmap: {selected_cat_col} per Cluster"
            )
            fig_heat.update_layout(height=400)
            st.plotly_chart(fig_heat, use_container_width=True)
        
        # 2. Insight Otomatis
        st.markdown("##### 🎯 Insight Otomatis per Cluster")
        
        insights = []
        for cluster_num in sorted(df['Cluster'].unique()):
            cluster_data = crosstab_pct.loc[cluster_num]
            dominant_category = cluster_data.idxmax()
            dominant_percentage = cluster_data.max()
            
            # Hitung diversity (berapa banyak kategori yang signifikan >10%)
            diverse_categories = (cluster_data > 10).sum()
            diversity_level = "Tinggi" if diverse_categories >= 3 else "Sedang" if diverse_categories == 2 else "Rendah"
            
            insights.append({
                'cluster': cluster_num,
                'dominant': dominant_category,
                'percentage': dominant_percentage,
                'diversity': diversity_level,
                'diverse_count': diverse_categories
            })
        
        # Tampilkan insights dalam cards
        cols = st.columns(min(k_value, 4))
        for idx, insight in enumerate(insights):
            with cols[idx % 4]:
                color = "#10B981" if insight['percentage'] > 50 else "#F59E0B" if insight['percentage'] > 30 else "#EF4444"
                
                # ✅ BENAR - Semua HTML dalam satu st.markdown()
                st.markdown(f"""
                <div style='
                    padding: 15px; 
                    border-radius: 10px; 
                    background: linear-gradient(135deg, {color}10 0%, {color}20 100%);
                    border-left: 4px solid {color};
                    margin-bottom: 10px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                '>
                    <h4 style='margin: 0 0 10px 0; color: #1E293B;'>Cluster {insight['cluster']}</h4>
                    <p style='margin: 5px 0; color: {color}; font-weight: 600;'>{insight['dominant']}</p>
                    <p style='margin: 5px 0; color: #475569;'><strong>{insight['percentage']:.1f}%</strong> dominan</p>
                    <p style='margin: 5px 0; color: #64748B; font-size: 12px;'>Diversitas: {insight['diversity']}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # 3. Tabel Detail
        st.markdown("##### 📋 Tabel Detail")
        
        tab1, tab2 = st.tabs(["Persentase (%)", "Jumlah Absolut"])
        
        with tab1:
            st.dataframe(
                crosstab_pct.style.background_gradient(cmap='Blues', axis=1).format("{:.1f}%"),
                use_container_width=True,
                height=300
            )
        
        with tab2:
            st.dataframe(
                crosstab_count,
                use_container_width=True,
                height=300
            )
        
        # 4. Analisis Multi-Kategorikal (jika ada lebih dari 1 kolom)
        if len(categorical_cols) > 1:
            st.markdown("---")
            st.markdown("#### 🔗 Analisis Multi-Kategorikal")
            
            col1, col2 = st.columns(2)
            
            with col1:
                cat_col_1 = st.selectbox(
                    "Pilih Kolom Pertama:",
                    categorical_cols,
                    key='multi_cat_1',
                    index=0
                )
            
            with col2:
                # Hapus kolom pertama dari pilihan kedua
                remaining_cols = [c for c in categorical_cols if c != cat_col_1]
                cat_col_2 = st.selectbox(
                    "Pilih Kolom Kedua:",
                    remaining_cols,
                    key='multi_cat_2',
                    index=0 if len(remaining_cols) > 0 else None
                )
            
            if cat_col_1 and cat_col_2:
                st.markdown(f"##### Analisis: {cat_col_1} vs {cat_col_2}")
                
                # Pivot table untuk kombinasi dua kategori
                pivot_data = df.groupby(['Cluster', cat_col_1, cat_col_2]).size().reset_index(name='Count')
                
                # Pilih cluster untuk analisis detail
                selected_cluster = st.selectbox(
                    "Pilih Cluster untuk Analisis Detail:",
                    sorted(df['Cluster'].unique()),
                    format_func=lambda x: f"Cluster {x}",
                    key='multi_cluster_select'
                )
                
                # Filter data untuk cluster yang dipilih
                cluster_pivot = pivot_data[pivot_data['Cluster'] == selected_cluster]
                
                if len(cluster_pivot) > 0:
                    # Buat pivot table
                    pivot_table = cluster_pivot.pivot_table(
                        index=cat_col_1, 
                        columns=cat_col_2, 
                        values='Count', 
                        fill_value=0
                    )
                    
                    # Heatmap untuk kombinasi kategori
                    fig_multi = px.imshow(
                        pivot_table,
                        labels=dict(x=cat_col_2, y=cat_col_1, color="Jumlah Konten"),
                        color_continuous_scale="Viridis",
                        aspect="auto",
                        title=f"Cluster {selected_cluster}: {cat_col_1} vs {cat_col_2}"
                    )
                    fig_multi.update_layout(height=500)
                    st.plotly_chart(fig_multi, use_container_width=True)
                    
                    # Tabel detail
                    with st.expander("📊 Lihat Data Detail"):
                        st.dataframe(
                            cluster_pivot.sort_values('Count', ascending=False),
                            use_container_width=True
                        )
                else:
                    st.warning(f"Tidak ada data untuk kombinasi kategori ini di Cluster {selected_cluster}")
        
        # 5. Summary Statistics
        st.markdown("---")
        st.markdown("#### 📊 Statistik Summary")
        
        summary_stats = []
        for cat_value in df[selected_cat_col].unique():
            cat_data = df[df[selected_cat_col] == cat_value]
            
            # Hitung distribusi cluster untuk kategori ini
            cluster_dist = cat_data['Cluster'].value_counts(normalize=True) * 100
            
            summary_stats.append({
                'Kategori': cat_value,
                'Jumlah': len(cat_data),
                'Persentase Total': len(cat_data) / len(df) * 100,
                'Cluster Dominan': cluster_dist.idxmax(),
                '% di Cluster Dominan': cluster_dist.max()
            })
        
        summary_df = pd.DataFrame(summary_stats)
        st.dataframe(
            summary_df.style.background_gradient(subset=['Jumlah', 'Persentase Total'], cmap='YlOrRd')
                       .format({'Persentase Total': '{:.1f}%', '% di Cluster Dominan': '{:.1f}%'}),
            use_container_width=True
        )
        
        # 6. Rekomendasi berdasarkan profiling
        st.markdown("---")
        st.markdown("#### 💡 Rekomendasi Berdasarkan Profiling")
        
        # Analisis untuk rekomendasi
        recommendations = []
        
        for cluster_num in sorted(df['Cluster'].unique()):
            cluster_cat_data = df[df['Cluster'] == cluster_num][selected_cat_col]
            top_categories = cluster_cat_data.value_counts(normalize=True).head(3)
            
            if len(top_categories) > 0:
                top_cat = top_categories.index[0]
                top_percentage = top_categories.iloc[0] * 100
                
                # Generate recommendation based on dominance
                if top_percentage > 70:
                    rec_text = f"Cluster ini sangat homogen ({top_percentage:.0f}% {top_cat}). Fokus pada konten untuk segmen ini."
                    rec_type = "success"
                elif top_percentage > 40:
                    rec_text = f"Cluster didominasi {top_cat} ({top_percentage:.0f}%). Prioritaskan strategi untuk segmen ini."
                    rec_type = "info"
                else:
                    rec_text = f"Cluster cukup beragam. Pertimbangkan strategi multi-segment untuk berbagai kategori."
                    rec_type = "warning"
                
                recommendations.append({
                    'cluster': cluster_num,
                    'text': rec_text,
                    'type': rec_type
                })
        
        # Tampilkan rekomendasi
        for rec in recommendations:
            if rec['type'] == 'success':
                st.success(f"**Cluster {rec['cluster']}:** {rec['text']}")
            elif rec['type'] == 'info':
                st.info(f"**Cluster {rec['cluster']}:** {rec['text']}")
            else:
                st.warning(f"**Cluster {rec['cluster']}:** {rec['text']}")
    
    # 7. Export Data
    if selected_cat_col:
        st.markdown("---")
        st.markdown("#### 💾 Export Data")
        
        # Buat data untuk export
        export_data = pd.crosstab(
            df['Cluster'], 
            df[selected_cat_col], 
            margins=True,
            margins_name='Total'
        )
        
        csv = export_data.to_csv().encode('utf-8')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                label="📥 Download Crosstab (CSV)",
                data=csv,
                file_name=f"categorical_profiling_{selected_cat_col}_k{k_value}.csv",
                mime="text/csv",
                type="primary"
            )
        
        with col2:
            if st.button("📊 Generate Report Summary"):
                st.markdown("### 📄 Profiling Report Summary")
                
                report_text = f"""
                ## Profiling Kategorikal Report
                **Dataset:** TikTok Content Data
                **Total Rows:** {len(df):,}
                **Number of Clusters:** {k_value}
                **Categorical Column Analyzed:** {selected_cat_col}
                
                ### Key Insights:
                """
                
                for insight in insights:
                    report_text += f"""
                    **Cluster {insight['cluster']}:**
                    - Dominant Category: {insight['dominant']} ({insight['percentage']:.1f}%)
                    - Diversity Level: {insight['diversity']}
                    """
                
                st.text_area("Report Summary", report_text, height=300)