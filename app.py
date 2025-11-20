import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import plotly.express as px    
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Analisis Pengadaan Barang/Jasa Pemerintah",
    page_icon="",
    layout="wide"
)

def load_data():
    df = pd.read_csv('cleaned_datatender.csv')
    return df

def preprocess_data(df):
    """Preprocess data untuk machine learning"""
    df_ml = df.copy()
    
    # Filter hanya data yang relevan untuk ML
    df_ml = df_ml[df_ml['status_tender'].isin(['Selesai', 'Gagal/Batal'])]
    
    # Pilih fitur yang akan digunakan
    features = ['pagu', 'hps_cleaned', 'jenis_pengadaan', 'mtd_evaluasi', 
               'kualifikasi_paket', 'mtd_pemilihan', 'jenis_klpd', 'sumber_dana']
    
    # Encode categorical variables
    le = LabelEncoder()
    categorical_cols = ['jenis_pengadaan', 'mtd_evaluasi', 'kualifikasi_paket', 
                       'mtd_pemilihan', 'jenis_klpd', 'sumber_dana']
    
    for col in categorical_cols:
        df_ml[col] = le.fit_transform(df_ml[col].astype(str))
    
    # Handle missing values
    df_ml = df_ml.dropna(subset=features + ['status_tender'])
    
    # Target variable
    df_ml['status_encoded'] = le.fit_transform(df_ml['status_tender'])
    
    return df_ml, features

def train_model(df_ml, features):
    """Train Random Forest model"""
    X = df_ml[features]
    y = df_ml['status_encoded']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, X_train, X_test, y_train, y_test, y_pred, accuracy

def main():
    st.title("Dashboard Analisis Pengadaan Barang/Jasa Pemerintah")
    st.markdown("Analisis komprehensif Data Tender Kualifikasi Usaha Kecil 2023-2024")
    
    # Load data
    df = load_data()
    
    # Sidebar
    st.sidebar.image("lkpp_logo.png", use_container_width=True)

    st.sidebar.title("Navigasi")
    analysis_type = st.sidebar.selectbox(
        "Pilih Analisis:",
        [
            "Overview",
            "Analisis Geografis",
            "Jenis Pengadaan & Metode",
            "Tingkat Kegagalan",
            "Prediksi ML"
        ]
    )

    
    # Main content berdasarkan pilihan
    if analysis_type == "Overview":
        show_overview(df)
    elif analysis_type == "Analisis Geografis":
        show_geographical_analysis(df)
    elif analysis_type == "Jenis Pengadaan & Metode":
        show_procurement_analysis(df)
    elif analysis_type == "Tingkat Kegagalan":
        show_failure_analysis(df)
    elif analysis_type == "Prediksi ML":
        show_ml_prediction(df)

def show_overview(df):
    st.header("Overview Data Pengadaan")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_projects = len(df)
        st.metric("Total Proyek", f"{total_projects:,}")
    
    with col2:
        total_budget = df['pagu'].sum() / 1e12
        st.metric("Total Pagu (Triliun)", f"Rp {total_budget:.2f}T")
    
    with col3:
        success_rate = (df['status_tender'] == 'Selesai').mean() * 100
        st.metric("Tingkat Kesuksesan", f"{success_rate:.1f}%")
    
    with col4:
        avg_efficiency = (df['pagu'] - df['nilai_kontrak']).mean() / 1e6
        st.metric("Rata2 Efisiensi (Juta)", f"Rp {avg_efficiency:.1f}M")
    
    st.subheader("Distribusi Proyek per Jenis KLPD")

    # Pastikan kolom benar
    klpd_counts = (
        df['jenis_klpd']
        .value_counts()
        .reset_index()
    )

    # Ubah nama kolom agar konsisten dan kompatibel
    klpd_counts.columns = ['jenis_klpd', 'jumlah']

    # Donut chart
    fig = px.pie(
        klpd_counts,
        names='jenis_klpd',     # SEKARANG kolomnya pasti ada
        values='jumlah',        # kolom nilai
        hole=0.45,
        title='Distribusi Proyek per Jenis KLPD'
    )

    fig.update_traces(
        textinfo='percent+label',
        textposition='inside'
    )

    st.plotly_chart(fig, use_container_width=True)


def show_geographical_analysis(df):
    st.header("Analisis Geografis")

    col1, col2 = st.columns(2)

    # --- Kolom 1: Jumlah Proyek per Jenis KLPD ---
    with col1:
        st.subheader("Distribusi Proyek per Jenis KLPD")

        # Hitung jumlah proyek dan siapkan dataframe yang rapi
        klpd_counts = (
            df['jenis_klpd']
            .value_counts()
            .reset_index()
        )
        klpd_counts.columns = ['jenis_klpd', 'jumlah']

        # Bar chart
        fig_bar = px.bar(
            klpd_counts,
            x='jenis_klpd',
            y='jumlah',
            text='jumlah',
            title='Jumlah Proyek per Jenis KLPD'
        )

        fig_bar.update_traces(textposition='outside')
        fig_bar.update_layout(
            xaxis_title="Jenis KLPD",
            yaxis_title="Jumlah Proyek",
            margin=dict(t=40, l=20, r=20, b=80)
        )

        st.plotly_chart(fig_bar, use_container_width=True)

    # --- Kolom 2: Nilai Kontrak per Jenis KLPD ---
    with col2:
        st.subheader("Nilai Kontrak per Jenis KLPD")

        contract_by_klpd = (
            df.groupby('jenis_klpd', as_index=False)['nilai_kontrak']
            .sum()
            .sort_values('nilai_kontrak', ascending=False)
        )

        fig_pie = px.pie(
            contract_by_klpd,
            names='jenis_klpd',
            values='nilai_kontrak',
            hole=0.45,
            title='Distribusi Nilai Kontrak per Jenis KLPD'
        )

        fig_pie.update_traces(
            textinfo='percent+label',
            textposition='inside'
        )

        fig_pie.update_layout(
            margin=dict(t=40, l=20, r=20, b=20)
        )

        st.plotly_chart(fig_pie, use_container_width=True)


    
    # Top 10 KLPD dengan proyek terbanyak
    st.subheader("10 KLPD dengan Proyek Terbanyak")

    # Hitung jumlah proyek per KLPD
    top_klpd = (
        df['nama_klpd']
        .value_counts()
        .head(10)
        .sort_values(ascending=True)  # agar urutan horizontal rapi
    )

    # Membuat bar chart horizontal
    fig = px.bar(
        top_klpd,
        x=top_klpd.values,
        y=top_klpd.index,
        orientation='h',
        labels={'x': 'Jumlah Proyek', 'y': 'Nama KLPD'},
        title='10 KLPD dengan Proyek Terbanyak'
    )

    # Menampilkan angka jumlah proyek di ujung bar
    fig.update_traces(
        text=top_klpd.values,
        textposition='outside'
    )

    # Layout lebih rapi
    fig.update_layout(
        xaxis_title="Jumlah Proyek",
        yaxis_title="KLPD",
        title_x=0.5,
        margin=dict(l=80, r=40)
    )

    st.plotly_chart(fig, use_container_width=True)


def show_procurement_analysis(df):
    st.header("Analisis Jenis Pengadaan & Metode")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Distribusi Jenis Pengadaan")
        procurement_counts = df['jenis_pengadaan'].value_counts()
        fig = px.pie(procurement_counts, values=procurement_counts.values,
                    names=procurement_counts.index,
                    title='Distribusi Jenis Pengadaan')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Metode Evaluasi Terpopuler")
        method_counts = df['mtd_evaluasi'].value_counts()
        fig = px.bar(
            x=method_counts.values,
            y=method_counts.index,
            orientation='h',
            text=method_counts.values,
            title='Metode Evaluasi Terpopuler'
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(
            xaxis_title="Jumlah",
            yaxis_title="Metode Evaluasi",
            margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Success rate by method
    st.subheader("Tingkat Kesuksesan per Metode Evaluasi")
    success_by_method = (
        df.groupby('mtd_evaluasi')['status_tender']
        .apply(lambda x: (x == 'Selesai').mean() * 100)
        .sort_values(ascending=False)
    )
    fig = px.bar(
        x=success_by_method.index,
        y=success_by_method.values,
        text=success_by_method.round(2).astype(str) + '%',
        title='Tingkat Kesuksesan per Metode Evaluasi'
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(
        xaxis_title="Metode Evaluasi",
        yaxis_title="Persentase Sukses (%)",
        margin=dict(l=10, r=10, t=40, b=10)
    )
    st.plotly_chart(fig, use_container_width=True)


def show_failure_analysis(df):
    st.header("Analisis Tingkat Kegagalan")
    failure_analysis = df['status_tender'].value_counts()
    col1, col2 = st.columns(2)

    with col1:
        fig = px.pie(
            names=failure_analysis.index,
            values=failure_analysis.values,
            title='Distribusi Status Tender'
        )
        fig.update_traces(
            textinfo='label+value+percent',
            textfont_size=12
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Failure rate by procurement type
        failure_by_type = (
            df.groupby('jenis_pengadaan')['status_tender']
            .apply(lambda x: (x == 'Gagal/Batal').mean() * 100)
            .sort_values(ascending=False)
        )
        fig = px.bar(
            x=failure_by_type.index,
            y=failure_by_type.values,
            text=failure_by_type.round(2).astype(str) + '%',
            title='Tingkat Kegagalan per Jenis Pengadaan'
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(
            xaxis_title="Jenis Pengadaan",
            yaxis_title="Persentase Gagal (%)",
            margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)

    
    # Analisis penyebab potensial
    st.subheader("Karakteristik Proyek yang Gagal")
    failed_projects = df[df['status_tender'] == 'Gagal/Batal']
    
    if len(failed_projects) > 0:
        st.write(f"Jumlah proyek gagal: {len(failed_projects)}")
        
        # Common characteristics of failed projects
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Metode Evaluasi Proyek Gagal:**")
            failed_methods = failed_projects['mtd_evaluasi'].value_counts()
            st.dataframe(failed_methods)
        
        with col2:
            st.write("**Jenis Pengadaan Proyek Gagal:**")
            failed_procurement = failed_projects['jenis_pengadaan'].value_counts()
            st.dataframe(failed_procurement)

def show_ml_prediction(df):
    st.header("Prediksi Status Tender dengan Machine Learning")
    
    st.info("""
    Model Random Forest akan dilatih untuk memprediksi apakah suatu tender akan:
    - **Selesai** atau **Gagal/Batal**
    berdasarkan karakteristik proyek.
    """)
    
    # Preprocess data
    df_ml, features = preprocess_data(df)
    
    if len(df_ml) < 100:
        st.warning("Data tidak cukup untuk training model yang reliable.")
        return
    
    # Train model
    with st.spinner('Training model...'):
        model, X_train, X_test, y_train, y_test, y_pred, accuracy = train_model(df_ml, features)
    
    # Display results
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Performance Model")
        st.metric("Accuracy", f"{accuracy:.3f}")
        
        # Classification report
        st.subheader("Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df)
    
    with col2:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig = px.imshow(cm, text_auto=True,
                       labels=dict(x="Predicted", y="Actual", color="Count"),
                       x=['Gagal/Batal', 'Selesai'],
                       y=['Gagal/Batal', 'Selesai'])
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance
    st.subheader("Feature Importance")
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    fig = px.bar(
        feature_importance,
        x='importance',
        y='feature',
        orientation='h',
        text=feature_importance['importance'].round(4),
        title='Feature Importance dalam Prediksi'
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(
        xaxis_title="Importance",
        yaxis_title="Feature",
        margin=dict(l=10, r=10, t=40, b=10)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Prediction interface
    st.subheader("Prediksi Status Tender Baru")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        pagu = st.number_input("Pagu (Rp)", min_value=0, value=1000000000)
        hps = st.number_input("HPS (Rp)", min_value=0, value=900000000)
    
    with col2:
        jenis_pengadaan = st.selectbox("Jenis Pengadaan", df['jenis_pengadaan'].unique())
        mtd_evaluasi = st.selectbox("Metode Evaluasi", df['mtd_evaluasi'].unique())
    
    with col3:
        kualifikasi_paket = st.selectbox("Kualifikasi Paket", df['kualifikasi_paket'].unique())
        mtd_pemilihan = st.selectbox("Metode Pemilihan", df['mtd_pemilihan'].unique())
    
    if st.button("Prediksi Status Tender"):
        # Create input array for prediction
        input_data = np.array([[pagu, hps, 
                              list(df['jenis_pengadaan'].unique()).index(jenis_pengadaan),
                              list(df['mtd_evaluasi'].unique()).index(mtd_evaluasi),
                              list(df['kualifikasi_paket'].unique()).index(kualifikasi_paket),
                              list(df['mtd_pemilihan'].unique()).index(mtd_pemilihan),
                              0, 0]])  # dummy values for jenis_klpd and sumber_dana
        
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]
        
        result = "Selesai" if prediction == 1 else "Gagal/Batal"
        confidence = probability[prediction] * 100
        
        st.success(f"**Prediksi:** {result}")
        st.info(f"**Tingkat Kepercayaan:** {confidence:.1f}%")

if __name__ == "__main__":
    main()
