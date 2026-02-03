import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# --- KONFIGURASI ---
st.set_page_config(page_title="Air Quality Retraining System", layout="wide")

# Session State untuk menyimpan model dan data antar menu
if 'model' not in st.session_state:
    st.session_state.model = None
    st.session_state.le = None
    st.session_state.features = []
    st.session_state.acc = 0
    st.session_state.df_full = None

# --- FUNGSI CORE ---
def train_model(df):
    # Preprocessing
    drop_cols = ['tanggal', 'stasiun', 'critical', 'location']
    df_clean = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore').dropna()
    
    le = LabelEncoder()
    df_clean['categori'] = le.fit_transform(df_clean['categori'])
    
    X = df_clean.select_dtypes(include=[np.number]).drop(columns=['categori'], errors='ignore')
    y = df_clean['categori']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Simpan ke Session State
    st.session_state.model = rf
    st.session_state.le = le
    st.session_state.features = X.columns.tolist()
    st.session_state.acc = accuracy_score(y_test, rf.predict(X_test))
    st.session_state.df_full = df

# Load Data Awal (Jakarta) jika belum ada
if st.session_state.model is None:
    url = "https://raw.githubusercontent.com/Ranggard/Dataset/main/Quality_Air_Jakarta.csv"
    train_model(pd.read_csv(url))

# --- SIDEBAR NAVIGASI ---
st.sidebar.title("Navigasi Sistem")
menu = st.sidebar.radio("Pilih Menu:", 
    ["Home", "Hasil Latih Dataset", "Prediksi Jakarta", "Prediksi Kota Lain", "Upload & Retraining"])

# --- MENU 1: HOME ---
if menu == "Home":
    st.title("🏠 Home: Klasifikasi Pencemaran Udara")
    st.markdown("""
    Aplikasi ini dirancang untuk memprediksi klasifikasi kualitas udara menggunakan algoritma **Random Forest**. 
    Keunggulan sistem ini adalah kemampuan **Re-training**, di mana pengguna dapat memperbarui model dengan 
    dataset terbaru dari berbagai kota untuk menjaga akurasi prediksi.
    """)
    
    st.subheader("📋 5 Data Teratas (Dataset Saat Ini)")
    st.dataframe(st.session_state.df_full.head(5), use_container_width=True)
    
    st.subheader("📊 Visualisasi Dataset")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Distribusi Kategori Udara**")
        fig1, ax1 = plt.subplots()
        sns.countplot(data=st.session_state.df_full, x='categori', palette='viridis', ax=ax1)
        plt.xticks(rotation=45)
        st.pyplot(fig1)
        
    with col2:
        st.write("**Rata-rata Nilai Polutan**")
        # Hanya ambil kolom angka untuk dirata-ratakan
        numeric_df = st.session_state.df_full.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            st.bar_chart(numeric_df.mean())

# --- MENU 2: HASIL LATIH ---
elif menu == "Hasil Latih Dataset":
    st.title("📈 Performa Pelatihan Model")
    
    # Statistik Utama
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Akurasi Pelatihan", f"{st.session_state.acc * 100:.2f}%")
    with col_b:
        st.metric("Jumlah Fitur Digunakan", len(st.session_state.features))

    st.divider()
    
    st.subheader("🎯 Visualisasi Pentingnya Fitur (Feature Importance)")
    st.write("Grafik di bawah menunjukkan polutan mana yang paling dominan dalam menentukan hasil klasifikasi.")
    
    importances = st.session_state.model.feature_importances_
    feat_df = pd.DataFrame({'Polutan': st.session_state.features, 'Kepentingan': importances})
    feat_df = feat_df.sort_values(by='Kepentingan', ascending=False)
    
    st.bar_chart(feat_df.set_index('Polutan'))
    
    st.info("Fitur dengan nilai tertinggi adalah kontributor utama dalam menentukan apakah udara masuk kategori Baik, Sedang, atau Tidak Sehat.")

# --- MENU 3 & 4: PREDIKSI (Sama seperti sebelumnya namun menggunakan session_state) ---
elif menu in ["Prediksi Jakarta", "Prediksi Kota Lain"]:
    title = "🏙️ Prediksi Jakarta" if menu == "Prediksi Jakarta" else "🌍 Prediksi Kota Lain"
    st.title(title)
    if menu == "Prediksi Kota Lain":
        st.warning("Catatan: Akurasi mungkin berbeda jika model belum dilatih dengan data kota ini.")
        
    with st.form("form_prediksi"):
        inputs = [st.number_input(f"Masukkan nilai {f}", value=0.0) for f in st.session_state.features]
        submit = st.form_submit_button("Klasifikasikan")
        
        if submit:
            pred = st.session_state.model.predict([inputs])
            label = st.session_state.le.inverse_transform(pred)
            st.success(f"Hasil Klasifikasi: **{label[0]}**")

# --- MENU 5: RETRAINING ---
elif menu == "Upload & Retraining":
    st.title("⚙️ Re-training Model")
    st.write("Gunakan menu ini untuk mengunggah dataset kota lain agar model dapat beradaptasi.")
    
    file = st.file_uploader("Upload File CSV", type="csv")
    if file:
        new_df = pd.read_csv(file)
        if st.button("Latih Ulang Sekarang"):
            with st.spinner("Proses Training..."):
                train_model(new_df)
                st.success("Model Berhasil Diperbarui! Silahkan cek menu Home atau Hasil Latih.")
