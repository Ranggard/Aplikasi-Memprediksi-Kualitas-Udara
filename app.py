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

# Rentang Validasi Wajar (Referensi ISPU Indonesia)
VALID_RANGES = {
    'pm10': (0, 600),
    'pm25': (0, 500),
    'so2': (0, 1000),
    'co': (0, 100),
    'o3': (0, 1000),
    'no2': (0, 1000)
}

# Session State
if 'model' not in st.session_state:
    st.session_state.model = None
    st.session_state.le = None
    st.session_state.features = []
    st.session_state.acc = 0
    st.session_state.df_full = None

def train_model(df):
    drop_cols = ['tanggal', 'stasiun', 'critical', 'location']
    df_clean = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore').dropna()
    le = LabelEncoder()
    df_clean['categori'] = le.fit_transform(df_clean['categori'])
    X = df_clean.select_dtypes(include=[np.number]).drop(columns=['categori'], errors='ignore')
    y = df_clean['categori']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    st.session_state.model = rf
    st.session_state.le = le
    st.session_state.features = X.columns.tolist()
    st.session_state.acc = accuracy_score(y_test, rf.predict(X_test))
    st.session_state.df_full = df

if st.session_state.model is None:
    url = "https://raw.githubusercontent.com/Ranggard/Dataset/main/Quality_Air_Jakarta.csv"
    train_model(pd.read_csv(url))

# --- SIDEBAR ---
st.sidebar.title("Navigasi Sistem")
menu = st.sidebar.radio("Pilih Menu:", ["Home", "Hasil Latih Dataset", "Prediksi Jakarta", "Prediksi Kota Lain", "Upload & Retraining"])

if menu == "Home":
    st.title("🏠 Home: Klasifikasi Pencemaran Udara")
    st.subheader("📋 5 Data Teratas (Dataset Saat Ini)")
    st.dataframe(st.session_state.df_full.head(5), use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Distribusi Kategori Udara**")
        fig1, ax1 = plt.subplots()
        sns.countplot(data=st.session_state.df_full, x='categori', palette='viridis', ax=ax1)
        st.pyplot(fig1)
    with col2:
        st.write("**Rata-rata Nilai Polutan**")
        numeric_df = st.session_state.df_full.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            st.bar_chart(numeric_df.mean())

elif menu == "Hasil Latih Dataset":
    st.title("📈 Performa Pelatihan Model")
    col_a, col_b = st.columns(2)
    col_a.metric("Akurasi Pelatihan", f"{st.session_state.acc * 100:.2f}%")
    col_b.metric("Jumlah Fitur", len(st.session_state.features))
    st.divider()
    feat_df = pd.DataFrame({'Polutan': st.session_state.features, 'Kepentingan': st.session_state.model.feature_importances_}).sort_values(by='Kepentingan', ascending=False)
    st.bar_chart(feat_df.set_index('Polutan'))

elif menu in ["Prediksi Jakarta", "Prediksi Kota Lain"]:
    st.title("🏙️ Prediksi" if menu == "Prediksi Jakarta" else "🌍 Prediksi Kota Lain")
    
    with st.form("form_prediksi"):
        input_values = []
        is_valid = True
        
        # Grid layout untuk input agar rapi
        cols = st.columns(len(st.session_state.features))
        for i, f in enumerate(st.session_state.features):
            with cols[i]:
                # Menggunakan text_input agar ada placeholder dan tanpa tombol +/-
                val = st.text_input(f"Nilai {f.upper()}", placeholder=f"Contoh: 50")
                
                # Validasi Angka & Rentang
                if val:
                    try:
                        f_val = float(val)
                        # Cek range wajar
                        low, high = VALID_RANGES.get(f.lower(), (0, 1000))
                        if f_val < low or f_val > high:
                            st.warning(f"⚠️ {f.upper()} Ekstrim ({f_val})")
                        input_values.append(f_val)
                    except ValueError:
                        st.error(f"❌ {f.upper()} harus angka")
                        is_valid = False
                else:
                    input_values.append(0.0) # Default jika kosong

        submit = st.form_submit_button("Klasifikasikan")
        
        if submit:
            if is_valid:
                pred = st.session_state.model.predict([input_values])
                label = st.session_state.le.inverse_transform(pred)
                st.success(f"Hasil Klasifikasi: **{label[0]}**")
            else:
                st.error("Mohon perbaiki input data sebelum memproses.")

elif menu == "Upload & Retraining":
    st.title("⚙️ Re-training Model")
    file = st.file_uploader("Upload File CSV", type="csv")
    if file:
        new_df = pd.read_csv(file)
        if st.button("Latih Ulang Sekarang"):
            with st.spinner("Proses Training..."):
                train_model(new_df)
                st.success("Model Berhasil Diperbarui!")
