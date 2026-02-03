import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import requests
from io import StringIO

# Konfigurasi Halaman
st.set_page_config(page_title="Air Pollution Classifier", layout="wide")

# URL Dataset Jakarta
DATA_URL = "https://raw.githubusercontent.com/Ranggard/Dataset/main/Quality_Air_Jakarta.csv"

@st.cache_data
def load_data(url):
    df = pd.read_csv(url)
    # Lakukan preprocessing dasar di sini (sesuaikan dengan kolom dataset)
    df = df.dropna()
    return df

def train_model(data):
    # Asumsikan kolom target adalah 'category' dan fitur lainnya adalah polutan
    # Sesuaikan nama kolom dengan dataset asli
    le = LabelEncoder()
    data['categori'] = le.fit_transform(data['categori'])
    
    X = data.drop(columns=['categori', 'tanggal', 'stasiun'], errors='ignore')
    y = data['categori']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    accuracy = rf.score(X_test, y_test)
    return rf, le, accuracy, X.columns

# Sidebar Menu
menu = st.sidebar.selectbox("Menu Utama", ["Dashboard", "Prediksi Jakarta", "Prediksi Kota Lain", "Re-training Model"])

df_jakarta = load_data(DATA_URL)
model, encoder, initial_acc, feature_names = train_model(df_jakarta)

if menu == "Dashboard":
    st.title("Dashboard Kualitas Udara")
    st.write("Statistik Dataset Jakarta (Dataset Rujukan)")
    st.dataframe(df_jakarta.head())
    # [Visualisasi 3: Bar Chart Distribusi Kategori Udara]
    st.bar_chart(df_jakarta['categori'].value_counts())

elif menu == "Prediksi Jakarta":
    st.header("Prediksi Kualitas Udara - Wilayah Jakarta")
    st.info(f"Model saat ini dilatih dengan data Jakarta. Akurasi: {initial_acc*100:.2f}%")
    
    # Input fitur berdasarkan dataset
    inputs = {}
    for col in feature_names:
        inputs[col] = st.number_input(f"Masukkan nilai {col}", value=0.0)
    
    if st.button("Klasifikasikan"):
        pred = model.predict([list(inputs.values())])
        res = encoder.inverse_transform(pred)
        st.success(f"Hasil Klasifikasi: {res[0]}")

elif menu == "Prediksi Kota Lain":
    st.header("Prediksi Kualitas Udara - Kota Lainnya")
    st.warning("Catatan: Akurasi mungkin berkurang karena model belum dilatih khusus untuk data kota ini.")
    
    # Input fitur yang sama
    inputs = {}
    for col in feature_names:
        inputs[col] = st.number_input(f"Masukkan nilai {col} (Data Kota Lain)", value=0.0)
    
    if st.button("Prediksi Sekarang"):
        pred = model.predict([list(inputs.values())])
        res = encoder.inverse_transform(pred)
        st.info(f"Hasil Prediksi Sementara: {res[0]}")

elif menu == "Re-training Model":
    st.header("Upload Dataset Baru & Re-training")
    st.write("Gunakan fitur ini jika Anda memiliki dataset dari kota lain untuk meningkatkan akurasi.")
    
    uploaded_file = st.file_uploader("Pilih file CSV", type="csv")
    if uploaded_file is not None:
        new_df = pd.read_csv(uploaded_file)
        st.write("Preview Data Baru:")
        st.dataframe(new_df.head())
        
        if st.button("Mulai Re-training"):
            with st.spinner('Melatih ulang model...'):
                new_model, new_encoder, new_acc, _ = train_model(new_df)
                st.success(f"Model Berhasil Diperbarui! Akurasi Baru: {new_acc*100:.2f}%")
                # Simpan model baru ke session state agar bisa digunakan di menu lain
                st.session_state['model'] = new_model
