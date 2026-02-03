import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. KONFIGURASI & STANDAR ISPU ---
st.set_page_config(page_title="Air Quality System", layout="wide")

# Batas validasi keras untuk menghindari data ekstrim (Input Ditolak)
RULES = {
    'pm10': (0, 500), 'pm25': (0, 500), 'so2': (0, 800),
    'co': (0, 100), 'o3': (0, 600), 'no2': (0, 800)
}

# Urutan Kategori Standar ISPU Indonesia
KATEGORI_ISPU = ['BAIK', 'SEDANG', 'TIDAK SEHAT', 'SANGAT TIDAK SEHAT', 'BERBAHAYA']

# --- 2. INISIALISASI STATE ---
if 'model' not in st.session_state:
    st.session_state.update({'model': None, 'le': None, 'features': [], 'acc': 0, 'df_full': None})

def train_model(df):
    try:
        # Pembersihan Label (Menyeragamkan ke 5 Kategori)
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

# Load Awal
if st.session_state.model is None:
    url = "https://raw.githubusercontent.com/Ranggard/Dataset/main/Quality_Air_Jakarta.csv"
    train_model(pd.read_csv(url))

# --- 3. NAVIGASI SIDEBAR ---
menu = st.sidebar.radio("Navigasi:", ["Home", "Hasil Latih Dataset", "Prediksi Jakarta", "Prediksi Kota Lain", "Upload & Retraining"])

# --- 4. LOGIKA MENU ---

if menu == "Home":
    st.title("🏠 Home: Analisis Kualitas Udara")
    st.markdown("""
    Sistem ini menggunakan algoritma **Random Forest** untuk menentukan klasifikasi udara (Baik, Sedang, Tidak Sehat, Sangat Tidak Sehat, Berbahaya).
    Fitur utama aplikasi ini adalah **Dynamic Retraining**, yang memungkinkan model belajar dari dataset baru 
    yang diunggah oleh pengguna, sehingga prediksi tetap akurat untuk berbagai kota.
    """)
    
    st.divider()    
    st.subheader("📋 Preview 5 Data Teratas")
    st.dataframe(st.session_state.df_full.head(5), use_container_width=True)
    
    st.divider()
    
    # Perbaikan: Mendefinisikan kolom sebelum digunakan
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Distribusi 5 Kategori ISPU**")
        fig1, ax1 = plt.subplots()
        # Filter kategori yang benar-benar ada di dataset dari 5 standar
        existing_cats = [c for c in KATEGORI_ISPU if c in st.session_state.df_full['categori'].unique()]
        sns.countplot(data=st.session_state.df_full, x='categori', order=existing_cats, palette='viridis', ax=ax1)
        plt.xticks(rotation=45)
        st.pyplot(fig1)
        
    with col2:
        st.write("**Rata-rata Konsentrasi Polutan**")
        numeric_df = st.session_state.df_full.select_dtypes(include=[np.number])
        st.bar_chart(numeric_df.mean())

elif menu == "Hasil Latih Dataset":
    st.title("📈 Hasil Latih Model")
    st.metric("Akurasi Model", f"{st.session_state.acc * 100:.2f}%")
    st.write("### Kepentingan Fitur (Feature Importance)")
    feat_df = pd.DataFrame({'Fitur': st.session_state.features, 'Value': st.session_state.model.feature_importances_}).sort_values(by='Value', ascending=False)
    st.bar_chart(feat_df.set_index('Fitur'))

elif menu in ["Prediksi Jakarta", "Prediksi Kota Lain"]:
    st.title(f"🔍 {menu}")
    with st.form("form_pred"):
        st.write("### Input Data Polutan")
        inputs = {}
        cols = st.columns(len(st.session_state.features))
        for i, f in enumerate(st.session_state.features):
            with cols[i]:
                inputs[f] = st.text_input(f.upper(), placeholder="Ketik angka...")
        
        if st.form_submit_button("Klasifikasikan"):
            errs, vals = [], []
            for f, v in inputs.items():
                try:
                    num = float(v)
                    low, high = RULES.get(f.lower(), (0, 1000))
                    if not (low <= num <= high): errs.append(f"{f.upper()} di luar batas ({low}-{high})")
                    else: vals.append(num)
                except: errs.append(f"{f.upper()} harus angka!")
            
            if errs: 
                for e in errs: st.error(e)
            else:
                p = st.session_state.model.predict([vals])
                st.success(f"Hasil: **{st.session_state.le.inverse_transform(p)[0]}**")

elif menu == "Upload & Retraining":
    st.title("⚙️ Re-training")
    file = st.file_uploader("Upload CSV", type="csv")
    if file and st.button("Proses Training"):
        if train_model(pd.read_csv(file)): st.success("Model diperbarui!")
        else: st.error("Format salah.")

