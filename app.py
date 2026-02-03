import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# --- INISIALISASI SESSION STATE ---
if 'model' not in st.session_state:
    st.session_state.model = None
    st.session_state.le = None
    st.session_state.features = []
    st.session_state.metrics = {}
    st.session_state.df_train = None

# --- FUNGSI PREPROCESSING & TRAINING ---
def train_process(df):
    # Cleaning
    drop_cols = ['tanggal', 'stasiun', 'critical', 'location']
    df_clean = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore').dropna()
    
    le = LabelEncoder()
    df_clean['categori'] = le.fit_transform(df_clean['categori'])
    
    X = df_clean.select_dtypes(include=[np.number]).drop(columns=['categori'], errors='ignore')
    y = df_clean['categori']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Simpan hasil ke session state
    y_pred = rf.predict(X_test)
    st.session_state.model = rf
    st.session_state.le = le
    st.session_state.features = X.columns.tolist()
    st.session_state.df_train = df
    st.session_state.metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'conf_matrix': confusion_matrix(y_test, y_pred),
        'report': classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True),
        'feat_importances': rf.feature_importances_
    }

# --- LOAD DATA AWAL (JAKARTA) ---
if st.session_state.model is None:
    URL_JKT = "https://raw.githubusercontent.com/Ranggard/Dataset/main/Quality_Air_Jakarta.csv"
    train_process(pd.read_csv(URL_JKT))

# --- SIDEBAR NAVIGASI ---
st.sidebar.title("Main Menu")
menu = st.sidebar.radio("Pilih Halaman:", 
    ["Beranda", "Hasil Latih Dataset", "Prediksi Jakarta", "Prediksi Kota Lain", "Upload & Retraining"])

# --- LOGIKA MENU ---

if menu == "Beranda":
    st.title("🍃 Air Quality Classifier")
    st.write("Selamat datang! Sistem ini menggunakan **Random Forest** untuk mengklasifikasikan tingkat pencemaran udara.")
    st.image("https://img.freepik.com/free-vector/city-with-air-pollution-concept_23-2148710771.jpg", width=500)
    st.write("Gunakan menu di samping untuk melihat hasil analisis atau melakukan prediksi.")

elif menu == "Hasil Latih Dataset":
    st.title("📈 Hasil Evaluasi Model")
    m = st.session_state.metrics
    
    # Metrik Utama
    col1, col2 = st.columns(2)
    col1.metric("Akurasi Model", f"{m['accuracy']*100:.2f}%")
    col2.metric("Jumlah Fitur", len(st.session_state.features))
    
    # Visualisasi 1: Confusion Matrix
    st.write("### Confusion Matrix")
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(m['conf_matrix'], annot=True, fmt='d', cmap='Blues', 
                xticklabels=st.session_state.le.classes_, yticklabels=st.session_state.le.classes_)
    plt.xlabel("Prediksi")
    plt.ylabel("Aktual")
    st.pyplot(fig)
    
    # Visualisasi 2: Feature Importance
    st.write("### Variabel Paling Berpengaruh (Feature Importance)")
    feat_df = pd.DataFrame({'Polutan': st.session_state.features, 'Importance': m['feat_importances']})
    st.bar_chart(feat_df.set_index('Polutan'))

elif menu == "Prediksi Jakarta":
    st.title("🏙️ Prediksi Wilayah Jakarta")
    st.info("Prediksi menggunakan dataset rujukan awal Jakarta.")
    
    inputs = [st.number_input(f"Nilai {f}", value=0.0) for f in st.session_state.features]
    if st.button("Proses Prediksi"):
        res = st.session_state.model.predict([inputs])
        st.success(f"Kualitas Udara: **{st.session_state.le.inverse_transform(res)[0]}**")

elif menu == "Prediksi Kota Lain":
    st.title("🌍 Prediksi Kota Lain")
    st.warning("Catatan: Gunakan menu 'Upload' jika ingin akurasi lebih tinggi untuk kota spesifik.")
    inputs = [st.number_input(f"Input {f}", key=f"other_{f}") for f in st.session_state.features]
    if st.button("Cek Prediksi"):
        res = st.session_state.model.predict([inputs])
        st.info(f"Hasil: **{st.session_state.le.inverse_transform(res)[0]}**")

elif menu == "Upload & Retraining":
    st.title("⚙️ Adaptasi Model Baru")
    u_file = st.file_uploader("Upload CSV Kota Lain", type="csv")
    if u_file:
        new_df = pd.read_csv(u_file)
        if st.button("Mulai Latih Ulang (Retrain)"):
            with st.spinner("Menyesuaikan model dengan data baru..."):
                train_process(new_df)
                st.balloons()
                st.success("Model berhasil diperbarui! Silahkan cek menu 'Hasil Latih Dataset'.")
