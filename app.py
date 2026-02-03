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
    except: 
        return False

# Load Awal (Menggunakan data Jakarta sebagai default)
if st.session_state.model is None:
    url = "https://raw.githubusercontent.com/Ranggard/Dataset/main/Quality_Air_Jakarta.csv"
    try:
        train_model(pd.read_csv(url))
    except:
        st.error("Gagal memuat dataset default. Silahkan unggah data di menu Retraining.")

# --- 3. NAVIGASI SIDEBAR MODERN ---
with st.sidebar:
    st.markdown("<h1 style='text-align: center; color: #1E88E5;'>AirNav System</h1>", unsafe_allow_html=True)
    st.markdown("---")
    menu = option_menu(
        menu_title="Main Menu", 
        options=["Home", "Hasil Latih", "Prediksi Jakarta", "Prediksi Kota Lain", "Retraining"],
        icons=["house", "bar-chart-line", "building", "geo-alt", "cloud-arrow-up"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "5!important", "background-color": "#fafafa"},
            "icon": {"color": "#1E88E5", "font-size": "20px"}, 
            "nav-link": {"font-size": "16px", "text-align": "left", "margin": "10px", "--hover-color": "#eee"},
            "nav-link-selected": {"background-color": "#1E88E5"},
        }
    )

# --- 4. LOGIKA MENU ---

if menu == "Home":
    st.title("🏠 Home: Analisis Kualitas Udara")
    st.markdown("""
    Sistem ini menggunakan algoritma **Random Forest** untuk klasifikasi pencemaran udara berdasarkan standar ISPU. 
    Dilengkapi dengan fitur **Dynamic Retraining** yang memungkinkan model beradaptasi dengan dataset baru secara instan.
    """)
    
    if st.session_state.df_full is not None:
        st.divider()    
        st.subheader("📋 Preview 5 Data Teratas")
        st.dataframe(st.session_state.df_full.head(5), use_container_width=True)
        
        st.divider()
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Distribusi 5 Kategori ISPU**")
            fig1, ax1 = plt.subplots()
            # Memastikan hanya kategori yang ada di dataset yang muncul di plot
            existing_cats = [c for c in KATEGORI_ISPU if c in st.session_state.df_full['categori'].unique()]
            sns.countplot(data=st.session_state.df_full, x='categori', order=existing_cats, palette='viridis', ax=ax1)
            plt.xticks(rotation=45)
            st.pyplot(fig1)
            
        with col2:
            st.write("**Rata-rata Konsentrasi Polutan**")
            numeric_df = st.session_state.df_full.select_dtypes(include=[np.number])
            if not numeric_df.empty:
                st.bar_chart(numeric_df.mean())
    else:
        st.warning("Data belum tersedia. Silahkan lakukan retraining.")

elif menu == "Hasil Latih":
    st.title("📈 Hasil Latih Model")
    if st.session_state.model is not None:
        st.metric("Akurasi Model", f"{st.session_state.acc * 100:.2f}%")
        st.divider()
        st.write("### Kepentingan Fitur (Feature Importance)")
        st.info("Grafik ini menunjukkan polutan mana yang paling berpengaruh dalam prediksi model.")
        feat_df = pd.DataFrame({
            'Fitur': st.session_state.features, 
            'Value': st.session_state.model.feature_importances_
        }).sort_values(by='Value', ascending=False)
        st.bar_chart(feat_df.set_index('Fitur'))
    else:
        st.error("Model belum dilatih.")

elif menu in ["Prediksi Jakarta", "Prediksi Kota Lain"]:
    st.title(f"🔍 {menu}")
    if st.session_state.model is not None:
        with st.form("form_pred"):
            st.write("### Input Data Polutan")
            st.write("Masukkan nilai konsentrasi polutan (Placeholder menunjukkan contoh input):")
            
            inputs = {}
            # Mengatur kolom agar input berjejer rapi
            cols = st.columns(len(st.session_state.features))
            for i, f in enumerate(st.session_state.features):
                with cols[i]:
                    inputs[f] = st.text_input(f.upper(), placeholder="0.0", key=f"input_{menu}_{f}")
            
            submitted = st.form_submit_button("Klasifikasikan")
            
            if submitted:
                errs, vals = [], []
                for f, v in inputs.items():
                    if v.strip() == "":
                        errs.append(f"{f.upper()} tidak boleh kosong.")
                    else:
                        try:
                            num = float(v)
                            low, high = RULES.get(f.lower(), (0, 1000))
                            if not (low <= num <= high): 
                                errs.append(f"{f.upper()} di luar batas wajar ({low}-{high})")
                            else: 
                                vals.append(num)
                        except ValueError: 
                            errs.append(f"{f.upper()} harus berupa angka!")
                
                if errs: 
                    for e in errs: st.error(e)
                else:
                    prediction = st.session_state.model.predict([vals])
                    label = st.session_state.le.inverse_transform(prediction)[0]
                    st.success(f"### Hasil Klasifikasi: **{label}**")
    else:
        st.error("Model tidak tersedia untuk prediksi.")

elif menu == "Retraining":
    st.title("⚙️ Re-training Model")
    st.write("Gunakan menu ini untuk memperbarui model dengan dataset baru (Format .CSV)")
    
    file = st.file_uploader("Pilih File CSV", type="csv")
    if file:
        if st.button("Mulai Proses Training"):
            with st.spinner("Sedang melatih ulang model..."):
                if train_model(pd.read_csv(file)): 
                    st.success("Berhasil! Model dan visualisasi telah diperbarui sesuai dataset baru.")
                    st.balloons()
                else: 
                    st.error("Format CSV tidak sesuai atau data 'categori' tidak ditemukan.")
