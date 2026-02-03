import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. KONFIGURASI & RENTANG VALIDASI KETAT ---
st.set_page_config(page_title="Sistem Prediksi Kualitas Udara", layout="wide")

# Batas angka wajar (Jika di luar ini, input ditolak demi keamanan data)
# Berdasarkan ambang batas ekstrim ISPU
RULES = {
    'pm10': (0, 500),
    'pm25': (0, 500),
    'so2': (0, 800),
    'co': (0, 100),
    'o3': (0, 600),
    'no2': (0, 800)
}

# --- 2. INISIALISASI SESSION STATE ---
if 'model' not in st.session_state:
    st.session_state.model = None
    st.session_state.le = None
    st.session_state.features = []
    st.session_state.acc = 0
    st.session_state.df_full = None

# --- 3. FUNGSI PELATIHAN MODEL ---
def train_model(df):
    try:
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
        return True
    except:
        return False

# Load data awal Jakarta
if st.session_state.model is None:
    url = "https://raw.githubusercontent.com/Ranggard/Dataset/main/Quality_Air_Jakarta.csv"
    train_model(pd.read_csv(url))

# --- 4. SIDEBAR NAVIGASI ---
st.sidebar.title("Navigasi Sistem")
menu = st.sidebar.radio("Pilih Menu:", ["Home", "Hasil Latih Dataset", "Prediksi Jakarta", "Prediksi Kota Lain", "Upload & Retraining"])

# --- MENU 1: HOME ---
if menu == "Home":
    st.title("🏠 Home: Klasifikasi Pencemaran Udara")

    st.markdown("""
    **Selamat Datang di Aplikasi Prediksi Kualitas Udara.**
    Sistem ini menggunakan algoritma **Random Forest** untuk menentukan klasifikasi udara (Baik, Sedang, Tidak Sehat).
    Fitur utama aplikasi ini adalah **Dynamic Retraining**, yang memungkinkan model belajar dari dataset baru 
    yang diunggah oleh pengguna, sehingga prediksi tetap akurat untuk berbagai kota.
    """)    
    
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
        st.bar_chart(numeric_df.mean())

# --- MENU 2: HASIL LATIH ---
elif menu == "Hasil Latih Dataset":
    st.title("📈 Performa Pelatihan Model")
    col_a, col_b = st.columns(2)
    col_a.metric("Akurasi Pelatihan", f"{st.session_state.acc * 100:.2f}%")
    col_b.metric("Jumlah Fitur", len(st.session_state.features))
    
    st.divider()
    st.subheader("🎯 Kepentingan Fitur (Feature Importance)")
    feat_df = pd.DataFrame({'Polutan': st.session_state.features, 'Kepentingan': st.session_state.model.feature_importances_}).sort_values(by='Kepentingan', ascending=False)
    st.bar_chart(feat_df.set_index('Polutan'))

# --- MENU 3 & 4: PREDIKSI (VALIDASI KETAT) ---
elif menu in ["Prediksi Jakarta", "Prediksi Kota Lain"]:
    st.title(f"🔍 {menu}")
    
    with st.form("form_prediksi"):
        st.write("### Masukkan Parameter Polutan")
        input_dict = {}
        
        # Grid layout untuk input
        cols = st.columns(len(st.session_state.features))
        for i, f in enumerate(st.session_state.features):
            with cols[i]:
                # text_input tanpa tombol +/- dan menggunakan placeholder
                user_val = st.text_input(f.upper(), placeholder="Mulai ketik...", key=f"input_{f}")
                input_dict[f] = user_val

        submitted = st.form_submit_button("Klasifikasikan")

        if submitted:
            final_values = []
            errors = []
            
            for feat, val in input_dict.items():
                if val.strip() == "":
                    errors.append(f"Kolom {feat.upper()} tidak boleh kosong.")
                else:
                    try:
                        num_val = float(val)
                        # Validasi Rentang Keras
                        low, high = RULES.get(feat.lower(), (0, 1000))
                        if num_val < low or num_val > high:
                            errors.append(f"NILAI DITOLAK: {feat.upper()} ({num_val}) di luar batas wajar ({low}-{high})!")
                        else:
                            final_values.append(num_val)
                    except ValueError:
                        errors.append(f"INPUT DITOLAK: {feat.upper()} harus berupa angka!")

            if errors:
                for err in errors:
                    st.error(err)
                st.warning("⚠️ Prediksi gagal diproses karena kesalahan input data.")
            else:
                pred = st.session_state.model.predict([final_values])
                label = st.session_state.le.inverse_transform(pred)
                st.success(f"### Hasil Klasifikasi: {label[0]}")

# --- MENU 5: RETRAINING ---
elif menu == "Upload & Retraining":
    st.title("⚙️ Re-training Model")
    file = st.file_uploader("Upload File CSV Baru", type="csv")
    if file:
        new_df = pd.read_csv(file)
        if st.button("Latih Ulang Sekarang"):
            with st.spinner("Proses..."):
                if train_model(new_df):
                    st.success("Model Berhasil Diperbarui!")
                else:
                    st.error("Format CSV tidak sesuai.")

