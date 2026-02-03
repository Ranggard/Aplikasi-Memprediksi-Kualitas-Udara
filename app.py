import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# --- FUNGSI PREPROCESSING ---
def preprocess_data(df):
    # Hapus kolom non-numerik yang tidak berguna untuk perhitungan
    drop_cols = ['tanggal', 'stasiun', 'critical', 'location']
    df_clean = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    
    # Menghapus baris yang memiliki nilai kosong
    df_clean = df_clean.dropna()
    
    # Encoding Target (Ubah 'BAIK', 'SEDANG' jadi angka)
    le = LabelEncoder()
    if 'categori' in df_clean.columns:
        df_clean['categori'] = le.fit_transform(df_clean['categori'])
    else:
        st.error("Dataset harus memiliki kolom 'categori' sebagai target!")
        return None, None, None

    # Pisahkan Fitur (X) dan Target (y)
    # Hanya ambil kolom numerik untuk X
    X = df_clean.select_dtypes(include=[np.number]).drop(columns=['categori'], errors='ignore')
    y = df_clean['categori']
    
    return X, y, le

# --- SETUP UI ---
st.set_page_config(page_title="Air Quality Prediction", layout="wide")
st.title("Sistem Klasifikasi Pencemaran Udara - Random Forest")

# Sidebar
menu = st.sidebar.selectbox("Pilih Menu", 
    ["Data Jakarta (Baseline)", "Prediksi Kota Jakarta", "Prediksi Kota Lain", "Re-training Model"])

# Load Dataset Jakarta sebagai default model
URL_JAKARTA = "https://raw.githubusercontent.com/Ranggard/Dataset/main/Quality_Air_Jakarta.csv"

@st.cache_resource
def get_base_model():
    df = pd.read_csv(URL_JAKARTA)
    X, y, le = preprocess_data(df)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    acc = model.score(X, y) # Simplifikasi untuk baseline
    return model, le, X.columns.tolist(), acc

base_model, base_le, features, base_acc = get_base_model()

# --- LOGIKA MENU ---
if menu == "Data Jakarta (Baseline)":
    st.subheader("Dataset Rujukan Jakarta")
    df_jkt = pd.read_csv(URL_JAKARTA)
    st.write(df_jkt.head())
    
    # VISUALISASI 1: Distribusi Kategori
    st.write("### Distribusi Kategori Udara")
    fig, ax = plt.subplots()
    sns.countplot(data=df_jkt, x='categori', ax=ax)
    st.pyplot(fig)

elif menu == "Prediksi Kota Jakarta":
    st.subheader("Input Data Kualitas Udara Jakarta")
    st.info(f"Model menggunakan Akurasi Baseline: {base_acc*100:.2f}%")
    
    col1, col2 = st.columns(2)
    input_data = []
    with col1:
        for f in features[:len(features)//2]:
            val = st.number_input(f"Nilai {f}", value=0.0)
            input_data.append(val)
    with col2:
        for f in features[len(features)//2:]:
            val = st.number_input(f"Nilai {f}", value=0.0)
            input_data.append(val)
            
    if st.button("Cek Kualitas Udara"):
        res = base_model.predict([input_data])
        label = base_le.inverse_transform(res)
        st.success(f"Hasil Klasifikasi: **{label[0]}**")

elif menu == "Prediksi Kota Lain":
    st.subheader("Prediksi Wilayah Luar Jakarta")
    st.warning("Catatan: Akurasi mungkin lebih rendah karena pola data kota ini berbeda dengan Jakarta.")
    
    input_data = [st.number_input(f"Input {f}", key=f) for f in features]
    if st.button("Prediksi"):
        res = base_model.predict([input_data])
        label = base_le.inverse_transform(res)
        st.info(f"Hasil Prediksi untuk Kota Lain: **{label[0]}**")

elif menu == "Re-training Model":
    st.subheader("Otomatisasi Re-training")
    st.write("Upload dataset kota lain (format CSV) untuk menyesuaikan model.")
    
    file = st.file_uploader("Upload CSV", type="csv")
    if file:
        new_df = pd.read_csv(file)
        st.write("Preview Data Baru:", new_df.head(3))
        
        if st.button("Mulai Latih Ulang"):
            X_new, y_new, le_new = preprocess_data(new_df)
            if X_new is not None:
                new_model = RandomForestClassifier(n_estimators=100)
                new_model.fit(X_new, y_new)
                new_acc = new_model.score(X_new, y_new)
                st.success(f"Re-training Berhasil! Akurasi Model Baru: {new_acc*100:.2f}%")
                # Visualisasi 2: Feature Importance
                st.write("### Polutan Paling Berpengaruh (Feature Importance)")
                feat_imp = pd.Series(new_model.feature_importances_, index=X_new.columns)
                st.bar_chart(feat_imp)
