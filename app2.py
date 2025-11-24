import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Konfigurasi halaman
st.set_page_config(
    page_title="Dashboard Analisis Tender Pemerintah",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('cleaned_datatender.csv')
    # Cleaning tambahan
    df['pagu'] = pd.to_numeric(df['pagu'], errors='coerce')
    df['hps_cleaned'] = pd.to_numeric(df['hps_cleaned'], errors='coerce')
    df['nilai_penawaran'] = pd.to_numeric(df['nilai_penawaran'], errors='coerce')
    df['nilai_kontrak'] = pd.to_numeric(df['nilai_kontrak'], errors='coerce')
    df['persentase_penurunan_hps'] = pd.to_numeric(df['persentase_penurunan_hps'], errors='coerce')
    
    # Filter hanya data yang selesai
    df_selesai = df[df['status_tender'] == 'Selesai'].copy()
    
    return df, df_selesai

df, df_selesai = load_data()

# Sidebar
st.sidebar.image("lkpp_logo.png", use_container_width=True)
st.sidebar.title("Kontrol Dashboard")
st.sidebar.markdown("---")

# Filter data
jenis_klpd = st.sidebar.multiselect(
    "Pilih Jenis KLPD:",
    options=df['jenis_klpd'].unique(),
    default=df['jenis_klpd'].unique()
)

sumber_dana = st.sidebar.multiselect(
    "Pilih Sumber Dana:",
    options=df['sumber_dana'].unique(),
    default=df['sumber_dana'].unique()
)

status_tender = st.sidebar.multiselect(
    "Pilih Status Tender:",
    options=df['status_tender'].unique(),
    default=['Selesai']
)

# Filter data berdasarkan pilihan
df_filtered = df[
    (df['jenis_klpd'].isin(jenis_klpd)) &
    (df['sumber_dana'].isin(sumber_dana)) &
    (df['status_tender'].isin(status_tender))
]

# Header
st.title("Dashboard Analisis Tender Pemerintah 2023-2024")
st.markdown("---")

# Metrics Overview
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_tender = len(df_filtered)
    st.metric("Total Paket Tender", f"{total_tender:,}")

with col2:
    total_pagu = df_filtered['pagu'].sum() / 1e12
    st.metric("Total Pagu (Triliun Rp)", f"{total_pagu:.2f}")

with col3:
    avg_efisiensi = df_selesai['persentase_penurunan_hps'].mean()
    st.metric("Rata-rata Efisiensi", f"{avg_efisiensi:.2f}%")

with col4:
    success_rate = (len(df_selesai) / len(df[df['status_tender'].isin(['Selesai', 'Gagal/Batal'])]) * 100 )
    st.metric("Success Rate Tender", f"{success_rate:.1f}%")

# Tab utama
tab1, tab2, tab3 = st.tabs(["Overview", "Analisis Institusi", "Analisis Detil",])

with tab1:
    st.header("Overview Tender Pemerintah")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribusi jenis KLPD
        fig_klpd = px.pie(
            df_filtered, 
            names='jenis_klpd',
            title='Distribusi Berdasarkan Jenis KLPD',
            hole=0.4
        )
        st.plotly_chart(fig_klpd, use_container_width=True)
        
        # Top 10 KLPD berdasarkan pagu
        top_klpd = df_filtered.groupby('nama_klpd')['pagu'].sum().nlargest(10).reset_index()
        fig_top_klpd = px.bar(
            top_klpd,
            x='pagu',
            y='nama_klpd',
            orientation='h',
            title='Top 10 KLPD Berdasarkan Pagu',
            labels={'pagu': 'Total Pagu (Rp)', 'nama_klpd': 'KLPD'}
        )
        st.plotly_chart(fig_top_klpd, use_container_width=True)
    
    with col2:
        # Status tender
        fig_status = px.pie(
            df_filtered,
            names='status_tender',
            title='Distribusi Status Tender',
            hole=0.4
        )
        st.plotly_chart(fig_status, use_container_width=True)
        
        # Metode pemilihan
        fig_metode = px.bar(
            df_filtered['mtd_pemilihan'].value_counts().reset_index(),
            x='count',
            y='mtd_pemilihan',
            orientation='h',
            title='Distribusi Metode Pemilihan'
        )
        st.plotly_chart(fig_metode, use_container_width=True)

with tab2:
    st.header("Analisis Berdasarkan Institusi")
    
    institusi = st.selectbox(
        "Pilih Institusi:",
        options=df['nama_klpd'].unique()
    )
    
    df_institusi = df[df['nama_klpd'] == institusi]
    df_institusi_selesai = df_institusi[df_institusi['status_tender'] == 'Selesai']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(f"Total Paket - {institusi}", len(df_institusi))
    with col2:
        st.metric("Pagu Total", f"Rp {df_institusi['pagu'].sum():,.0f}")
    with col3:
        if len(df_institusi_selesai) > 0:
            efisiensi = df_institusi_selesai['persentase_penurunan_hps'].mean()
            st.metric("Rata-rata Efisiensi", f"{efisiensi:.2f}%")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Satker terbesar dalam institusi
        satker_pagu = df_institusi.groupby('nama_satker')['pagu'].sum().nlargest(10)
        fig_satker = px.bar(
            satker_pagu.reset_index(),
            x='pagu',
            y='nama_satker',
            orientation='h',
            title=f'Top 10 Satker Berdasarkan Pagu - {institusi}'
        )
        st.plotly_chart(fig_satker, use_container_width=True)
    
    with col2:
        # Jenis pengadaan di institusi
        fig_jenis = px.pie(
            df_institusi,
            names='jenis_pengadaan',
            title=f'Distribusi Jenis Pengadaan - {institusi}'
        )
        st.plotly_chart(fig_jenis, use_container_width=True)

with tab3:
    st.header("Analisis Detil dan Eksplorasi Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Filter interaktif berdasarkan rentang pagu
        min_pagu, max_pagu = st.slider(
            "Rentang Pagu:",
            min_value=float(df['pagu'].min()),
            max_value=float(df['pagu'].max()),
            value=(float(df['pagu'].min()), float(df['pagu'].max())),
            step=100000000.0
        )
        
        # Filter interaktif berdasarkan efisiensi
        min_efisiensi, max_efisiensi = st.slider(
            "Rentang Efisiensi (%):",
            min_value=float(df_selesai['persentase_penurunan_hps'].min()),
            max_value=float(df_selesai['persentase_penurunan_hps'].max()),
            value=(0.0, 25.0),
            step=0.5
        )
    
    with col2:
        jenis_pengadaan_detil = st.multiselect(
            "Pilih Jenis Pengadaan:",
            options=df['jenis_pengadaan'].unique(),
            default=df['jenis_pengadaan'].unique()
        )
        
        kualifikasi = st.multiselect(
            "Pilih Kualifikasi:",
            options=df['kualifikasi_paket'].unique(),
            default=df['kualifikasi_paket'].unique()
        )
    
    # Terapkan filter
    df_detil = df_selesai[
        (df_selesai['pagu'] >= min_pagu) &
        (df_selesai['pagu'] <= max_pagu) &
        (df_selesai['persentase_penurunan_hps'] >= min_efisiensi) &
        (df_selesai['persentase_penurunan_hps'] <= max_efisiensi) &
        (df_selesai['jenis_pengadaan'].isin(jenis_pengadaan_detil)) &
        (df_selesai['kualifikasi_paket'].isin(kualifikasi))
    ]
    
    st.subheader(f"Data Terfilter: {len(df_detil)} paket tender")
    
    # Tabel data
    st.dataframe(
        df_detil[['nama_paket', 'nama_klpd', 'pagu', 'nilai_kontrak', 'persentase_penurunan_hps']].head(20),
        use_container_width=True
    )
    
    # Download data terfilter
    csv = df_detil.to_csv(index=False)
    st.download_button(
        label="Download Data Terfilter",
        data=csv,
        file_name="data_tender_terfilter.csv",
        mime="text/csv"
    )

# Footer
st.markdown("---")
st.markdown(
    "Dashboard ini dibuat untuk analisis data tender pemerintah Indonesia tahun 2023-2024. "
    "Data bersumber dari LPSE berbagai institusi."
)