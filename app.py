import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Prediksi Kualitas Udara", layout="wide")

# --- FUNGSI PREPROCESSING ---
def clean_data(df):
    # Hapus kolom yang tidak diperlukan untuk model
    drop_cols = ['tanggal', 'stasiun', 'critical', 'location']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    df = df.dropna()
    
    # Label Encoding untuk target 'categori'
    le = LabelEncoder()
    if 'categori' in df.columns:
        df['categori'] = le.fit_transform(df['categori'])
        return df, le
    return None, None

# --- LOAD DATA AWAL (JAKARTA) ---
URL_JKT = "https://raw.githubusercontent.com/Ranggard/Dataset/main/Quality_Air_Jakarta.csv"

@st.cache_resource
def initial_training():
    df_raw = pd.read_csv(URL_JKT)
    df_clean, le = clean_data(df_raw)
    
    X = df_clean.drop(columns=['categori'])
    y = df_clean['categori']
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, le, X.columns.tolist(), df_raw

model, le, features, raw_data_jkt = initial_training()

# --- SIDEBAR MENU (RADIO BUTTONS) ---
st.sidebar.title("Navigasi")
menu = st.sidebar.radio(
    "Pilih Menu:",
    ["Beranda & Analisis Data", "Prediksi Kota Jakarta", "Prediksi Kota Lainnya", "Retraining Model"]
)

# --- LOGIKA MENU ---

if menu == "Beranda & Analisis Data":
    st.title("📊 Analisis Data Kualitas Udara (Baseline: Jakarta)")
    st.write("Visualisasi data historis Jakarta yang digunakan untuk pelatihan awal model.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("### 5 Data Teratas")
        st.dataframe(raw_data_jkt.head())
    
    with col2:
        st.write("### Distribusi Kategori")
        fig, ax = plt.subplots()
        sns.countplot(data=raw_data_jkt, x='categori', ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)

elif menu == "Prediksi Kota Jakarta":
    st.title("🏙️ Prediksi Wilayah Jakarta")
    st.info("Model menggunakan parameter yang sudah dioptimasi untuk data Jakarta.")
    
    inputs = {}
    cols = st.columns(3)
    for i, f in enumerate(features):
        with cols[i % 3]:
            inputs[f] = st.number_input(f"Masukkan {f}", value=0.0)
            
    if st.button("Klasifikasikan Sekarang"):
        input_array = np.array([list(inputs.values())])
        prediction = model.predict(input_array)
        hasil = le.inverse_transform(prediction)
        st.success(f"Hasil Prediksi: **{hasil[0]}**")

elif menu == "Prediksi Kota Lainnya":
    st.title("🌍 Prediksi Wilayah Lain")
    st.warning("⚠️ Perhatian: Akurasi mungkin tidak optimal karena model belum mengenal pola polusi di kota ini.")
    
    inputs = {}
    cols = st.columns(3)
    for i, f in enumerate(features):
        with cols[i % 3]:
            inputs[f] = st.number_input(f"Input {f} (Kota Lain)", value=0.0)
            
    if st.button("Prediksi"):
        input_array = np.array([list(inputs.values())])
        prediction = model.predict(input_array)
        hasil = le.inverse_transform(prediction)
        st.info(f"Hasil Prediksi Sementara: **{hasil[0]}**")

elif menu == "Retraining Model":
    st.title("⚙️ Re-training Model (Update Model)")
    st.write("Upload dataset baru untuk melatih ulang algoritma Random Forest agar sesuai dengan wilayah baru.")
    
    uploaded_file = st.file_uploader("Upload File CSV Kota Anda", type="csv")
    
    if uploaded_file is not None:
        new_df_raw = pd.read_csv(uploaded_file)
        st.write("Data Berhasil Diunggah:")
        st.dataframe(new_df_raw.head(3))
        
        if st.button("Mulai Proses Training Ulang"):
            with st.spinner("Sedang memproses..."):
                new_df_clean, new_le = clean_data(new_df_raw)
                if new_df_clean is not None:
                    X_new = new_df_clean.drop(columns=['categori'])
                    y_new = new_df_clean['categori']
                    
                    # Training ulang
                    new_model = RandomForestClassifier(n_estimators=100)
                    new_model.fit(X_new, y_new)
                    
                    st.success(f"Model Berhasil Diperbarui dengan data baru!")
                    
                    # Visualisasi Feature Importance
                    st.write("### Polutan Paling Berpengaruh di Wilayah Baru")
                    importances = new_model.feature_importances_
                    feat_df = pd.DataFrame({'Polutan': X_new.columns, 'Importance': importances})
                    st.bar_chart(feat_df.set_index('Polutan'))
