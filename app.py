import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Air Quality Retraining System", layout="wide")

# Inisialisasi Session State agar data tidak hilang saat pindah menu
if 'df_full' not in st.session_state:
    st.session_state.df_full = None
    st.session_state.model = None
    st.session_state.le = None
    st.session_state.features = []
    st.session_state.acc = 0

# --- FUNGSI PROSES DATA ---
def train_logic(df):
    # Bersihkan data
    drop_cols = ['tanggal', 'stasiun', 'critical', 'location']
    df_clean = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore').dropna()
    
    # Encode target
    le = LabelEncoder()
    df_clean['categori'] = le.fit_transform(df_clean['categori'])
    
    # Pisahkan X dan y
    X = df_clean.select_dtypes(include=[np.number]).drop(columns=['categori'], errors='ignore')
    y = df_clean['categori']
    
    # Split & Train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Simpan ke State
    st.session_state.model = rf
    st.session_state.le = le
    st.session_state.features = X.columns.tolist()
    st.session_state.acc = accuracy_score(y_test, rf.predict(X_test))
    st.session_state.df_full = df

# Load awal jika masih kosong (Baseline Jakarta)
if st.session_state.df_full is None:
    url = "https://raw.githubusercontent.com/Ranggard/Dataset/main/Quality_Air_Jakarta.csv"
    train_logic(pd.read_csv(url))

# --- SIDEBAR MENU ---
st.sidebar.title("Menu Utama")
menu = st.sidebar.radio("Pilih Halaman:", ["Home", "Hasil Latih Dataset", "Prediksi Jakarta", "Prediksi Kota Lain", "Upload & Retraining"])

# --- HALAMAN 1: HOME ---
if menu == "Home":
    st.title("🏠 Home")
    st.markdown("""
    ### Sistem Klasifikasi Pencemaran Udara (Random Forest)
    Aplikasi ini dirancang untuk memprediksi tingkat kualitas udara berdasarkan parameter polutan. 
    Menggunakan pendekatan **Re-training**, model dapat diperbarui secara mandiri dengan dataset kota lain 
    untuk menjaga relevansi hasil prediksi di lokasi yang berbeda.
    """)
    
    st.divider()
    
    st.subheader("📋 Preview 5 Data Teratas")
    st.dataframe(st.session_state.df_full.head(5), use_container_width=True)
    
    st.subheader("📊 Visualisasi Dataset")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Distribusi Kategori Udara**")
        fig1, ax1 = plt.subplots()
        sns.countplot(data=st.session_state.df_full, x='categori', palette='viridis', ax=ax1)
        st.pyplot(fig1)
    with col2:
        st.write("**Rata-rata Konsentrasi Polutan**")
        num_cols = st.session_state.df_full.select_dtypes(include=[np.number])
        st.bar_chart(num_cols.mean())

# --- HALAMAN 2: HASIL LATIH ---
elif menu == "Hasil Latih Dataset":
    st.title("📈 Hasil Latih Dataset")
    
    # Metrik Akurasi
    st.metric("Akurasi Model", f"{st.session_state.acc * 100:.2f} %")
    
    st.divider()
    
    st.subheader("🚀 Visualisasi Feature Importance")
    st.write("Menampilkan polutan yang paling berpengaruh dalam pengambilan keputusan model Random Forest.")
    
    importances = st.session_state.model.feature_importances_
    feat_df = pd.DataFrame({'Polutan': st.session_state.features, 'Importance': importances})
    feat_df = feat_df.sort_values(by='Importance', ascending=False)
    
    st.bar_chart(feat_df.set_index('Polutan'))

# --- HALAMAN LAINNYA ---
elif menu in ["Prediksi Jakarta", "Prediksi Kota Lain"]:
    st.title(f"🔍 {menu}")
    if menu == "Prediksi Kota Lain":
        st.warning("Catatan: Akurasi mungkin berbeda sebelum dilakukan Re-training untuk kota ini.")
    
    with st.form("input_form"):
        inputs = [st.number_input(f"Nilai {f}", value=0.0) for f in st.session_state.features]
        if st.form_submit_button("Klasifikasikan"):
            res = st.session_state.model.predict([inputs])
            label = st.session_state.le.inverse_transform(res)
            st.success(f"Hasil Klasifikasi: **{label[0]}**")

elif menu == "Upload & Retraining":
    st.title("⚙️ Re-training Model")
    file = st.file_uploader("Upload CSV Baru", type="csv")
    if file:
        new_df = pd.read_csv(file)
        if st.button("Latih Ulang Sekarang"):
            with st.spinner("Proses..."):
                train_logic(new_df)
                st.success("Model berhasil diperbarui!")
