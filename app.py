# Library yang digunakan
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_option_menu import option_menu 

# --- 1. KONFIGURASI & STANDAR ISPU ---
st.set_page_config(page_title="Prediksi Kualitas Udara", layout="wide")

# Batas validasi untuk input (Hard Validation)
RULES = {
    'pm10': (0, 500), 'pm25': (0, 500), 'so2': (0, 800),
    'co': (0, 100), 'o3': (0, 600), 'no2': (0, 800)
}
KATEGORI_ISPU = ['BAIK', 'SEDANG', 'TIDAK SEHAT', 'SANGAT TIDAK SEHAT', 'BERBAHAYA']

# --- 2. INISIALISASI STATE ---
if 'model' not in st.session_state:
    st.session_state.update({'model': None, 'le': None, 'features': [], 'acc': 0, 'df_full': None})

def train_model(df):
    try:
        df['categori'] = df['categori'].str.upper().str.strip()
        drop_cols = ['tanggal', 'stasiun', 'critical', 'location']
        df_clean = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore').dropna()
        le = LabelEncoder()
        df_clean['categori'] = le.fit_transform(df_clean['categori'])
        X = df_clean.select_dtypes(include=[np.number]).drop(columns=['categori'], errors='ignore')
        y = df_clean['categori']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        st.session_state.update({
            'model': rf, 'le': le, 'features': X.columns.tolist(),
            'acc': accuracy_score(y_test, rf.predict(X_test)), 'df_full': df
        })
        return True
    except: return False

# Load Awal Otomatis
if st.session_state.model is None:
    url = "https://raw.githubusercontent.com/Ranggard/Dataset/main/Quality_Air_Jakarta.csv"
    try: train_model(pd.read_csv(url))
    except: st.error("Gagal memuat dataset default.")

# --- 3. SIDEBAR NAVIGASI MODERN ---
with st.sidebar:
    st.markdown("<br><h2 style='text-align: center; color: #1E88E5; font-family: sans-serif;'>Prediksi Kualitas Udara</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 0.85em; color: gray;'>Random Forest Classifier</p>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    menu = option_menu(
        menu_title=None, # Menghapus teks "Main Menu"
        options=["Home", "Hasil Latih", "Prediksi Jakarta", "Prediksi Kota Lain", "Retraining"],
        icons=["house-door", "graph-up-arrow", "geo-fill", "map", "cloud-upload"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "#1E88E5", "font-size": "18px"}, 
            "nav-link": {
                "font-size": "15px", 
                "text-align": "left", 
                "margin": "10px 0px", # Jarak antar menu lebih lega
                "--hover-color": "#f0f2f6",
                "font-family": "sans-serif"
            },
            "nav-link-selected": {"background-color": "#1E88E5", "color": "white"},
        }
    )

# --- 4. LOGIKA HALAMAN ---
if menu == "Home":
    st.title("🏠 Home")
    
    st.markdown("""
    Sistem ini menggunakan algoritma **Random Forest** untuk menentukan klasifikasi udara (Baik, Sedang, Tidak Sehat, Sangat Tidak Sehat, Berbahaya).
    Fitur utama aplikasi ini adalah **Dynamic Retraining**, yang memungkinkan model belajar dari dataset baru 
    yang diunggah oleh pengguna, sehingga prediksi tetap akurat untuk berbagai kota.
    """)
    
    if st.session_state.df_full is not None:
        st.divider()
        st.subheader("📋 Preview Data Teratas")
        st.dataframe(st.session_state.df_full.head(5), use_container_width=True)
        st.divider()
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Distribusi 5 Kategori ISPU**")
            fig1, ax1 = plt.subplots()
            existing_cats = [c for c in KATEGORI_ISPU if c in st.session_state.df_full['categori'].unique()]
            sns.countplot(data=st.session_state.df_full, x='categori', order=existing_cats, palette='viridis', ax=ax1)
            plt.xticks(rotation=45)
            st.pyplot(fig1)
        with col2:
            st.write("**Rata-rata Konsentrasi Polutan**")
            st.bar_chart(st.session_state.df_full.select_dtypes(include=[np.number]).mean())

elif menu == "Hasil Latih":
    st.title("📈 Hasil Latih Model")
    st.metric("Akurasi Model", f"{st.session_state.acc * 100:.2f}%")
    st.divider()
    
    feat_df = pd.DataFrame({
        'Fitur': st.session_state.features, 
        'Value': st.session_state.model.feature_importances_
    }).sort_values(by='Value', ascending=False)
    
    st.write("### Kepentingan Fitur (Feature Importance)")
    st.bar_chart(feat_df.set_index('Fitur'))

elif menu in ["Prediksi Jakarta", "Prediksi Kota Lain"]:
    st.title(f"🔍 {menu}")
    with st.form("form_pred"):
        st.write("### Input Data Polutan")
        inputs = {}
        cols = st.columns(len(st.session_state.features))
        for i, f in enumerate(st.session_state.features):
            with cols[i]:
                # text_input tanpa tombol +/- dan placeholder rapi
                inputs[f] = st.text_input(f.upper(), placeholder="0.0", key=f"in_{menu}_{f}")
        
        if st.form_submit_button("Klasifikasikan"):
            errs, vals = [], []
            for f, v in inputs.items():
                if v.strip() == "":
                    errs.append(f"{f.upper()} tidak boleh kosong.")
                else:
                    try:
                        num = float(v)
                        low, high = RULES.get(f.lower(), (0, 1000))
                        if not (low <= num <= high): 
                            errs.append(f"{f.upper()} di luar batas ({low}-{high})")
                        else: vals.append(num)
                    except: 
                        errs.append(f"{f.upper()} harus angka!")
            
            if errs: 
                for e in errs: st.error(e)
            else:
                p = st.session_state.model.predict([vals])
                label = st.session_state.le.inverse_transform(p)[0]
                st.success(f"### Hasil Klasifikasi: **{label}**")

elif menu == "Retraining":
    st.title("⚙️ Re-training Model")
    st.write("Upload file CSV untuk memperbarui model secara otomatis.")
    file = st.file_uploader("Pilih File CSV", type="csv")
    if file and st.button("Mulai Latih Ulang"):
        if train_model(pd.read_csv(file)): 
            st.success("Model Berhasil Diperbarui!")
        else: 
            st.error("Format data salah atau kolom 'categori' tidak ditemukan.")

