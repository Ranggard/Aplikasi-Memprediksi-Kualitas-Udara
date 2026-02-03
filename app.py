# ================== LIBRARY ==================
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_option_menu import option_menu 

# ================== KONFIGURASI ==================
st.set_page_config(page_title="Prediksi Kualitas Udara", layout="wide")

RULES = {
    'pm10': (0, 500), 'pm25': (0, 500), 'so2': (0, 800),
    'co': (0, 100), 'o3': (0, 600), 'no2': (0, 800)
}
KATEGORI_ISPU = ['BAIK', 'SEDANG', 'TIDAK SEHAT', 'SANGAT TIDAK SEHAT', 'BERBAHAYA']

# ================== SESSION STATE ==================
if 'model' not in st.session_state:
    st.session_state.update({
        'model': None,
        'le': None,
        'features': [],
        'acc': 0,
        'df_full': None,
        'y_test': None,
        'y_pred': None
    })

# ================== TRAINING FUNCTION ==================
def train_model(df):
    try:
        df['categori'] = df['categori'].str.upper().str.strip()

        drop_cols = ['tanggal', 'stasiun', 'critical', 'location']
        df_clean = df.drop(
            columns=[c for c in drop_cols if c in df.columns],
            errors='ignore'
        ).dropna()

        le = LabelEncoder()
        df_clean['categori'] = le.fit_transform(df_clean['categori'])

        X = df_clean.select_dtypes(include=[np.number]).drop(columns=['categori'], errors='ignore')
        y = df_clean['categori']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)

        y_pred = rf.predict(X_test)

        st.session_state.update({
            'model': rf,
            'le': le,
            'features': X.columns.tolist(),
            'acc': accuracy_score(y_test, y_pred),
            'df_full': df,
            'y_test': y_test,
            'y_pred': y_pred
        })
        return True
    except:
        return False

# ================== LOAD DATA DEFAULT ==================
if st.session_state.model is None:
    url = "https://raw.githubusercontent.com/Ranggard/Dataset/main/Quality_Air_Jakarta.csv"
    try:
        train_model(pd.read_csv(url))
    except:
        st.error("Gagal memuat dataset default.")

# ================== SIDEBAR ==================
with st.sidebar:
    st.markdown("<br><h2 style='text-align: center; color: #1E88E5;'>Prediksi Kualitas Udara</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 0.85em; color: gray;'>Random Forest Classifier</p>", unsafe_allow_html=True)

    menu = option_menu(
        menu_title=None,
        options=[
            "Home",
            "Hasil Latih",
            "Prediksi Jakarta",
            "Prediksi Kota Lain",
            "Retraining"
        ],
        icons=["house", "bar-chart", "geo", "map", "cloud-upload"],
        default_index=0
    )

# ================== HOME (TIDAK DIUBAH) ==================
if menu == "Home":
    st.title("🏠 Home")
    st.markdown("""
    Sistem ini menggunakan algoritma **Random Forest** untuk menentukan klasifikasi udara
    (Baik, Sedang, Tidak Sehat, Sangat Tidak Sehat, Berbahaya).
    Fitur utama aplikasi ini adalah **Retraining**, yang memungkinkan model
    belajar dari dataset baru yang diunggah pengguna.
    """)

# ================== HASIL LATIH ==================
elif menu == "Hasil Latih":
    st.title("📈 Hasil Latih Model")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Akurasi Model", f"{st.session_state.acc * 100:.2f}%")
    with col2:
        st.metric("Jumlah Fitur", len(st.session_state.features))

    st.write(
        "Akurasi menunjukkan proporsi prediksi yang benar terhadap seluruh data uji. "
        "Nilai akurasi yang tinggi menandakan model mampu mengklasifikasikan kualitas udara dengan baik."
    )

    # ===== Confusion Matrix =====
    st.subheader("🧩 Confusion Matrix")
    st.write(
        "Confusion Matrix digunakan untuk mengevaluasi kinerja model klasifikasi dengan "
        "membandingkan label aktual dan hasil prediksi. "
        "Semakin dominan nilai diagonal, semakin baik performa model."
    )

    cm = confusion_matrix(st.session_state.y_test, st.session_state.y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=st.session_state.le.classes_,
        yticklabels=st.session_state.le.classes_,
        ax=ax
    )
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Aktual")
    st.pyplot(fig)

    # ===== Feature Importance =====
    st.subheader("⭐ Feature Importance")
    st.write(
        "Visualisasi ini menunjukkan tingkat kontribusi masing-masing fitur polutan "
        "dalam proses pengambilan keputusan model Random Forest. "
        "Fitur dengan nilai importance lebih tinggi memiliki pengaruh lebih besar terhadap hasil klasifikasi."
    )

    feat_df = pd.DataFrame({
        "Fitur": st.session_state.features,
        "Importance": st.session_state.model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    st.bar_chart(feat_df.set_index("Fitur"))

# ================== PREDIKSI ==================
elif menu in ["Prediksi Jakarta", "Prediksi Kota Lain"]:
    st.title(f"🔍 {menu}")

    with st.form("form_pred"):
        inputs = {}
        cols = st.columns(len(st.session_state.features))

        for i, f in enumerate(st.session_state.features):
            with cols[i]:
                inputs[f] = st.text_input(f.upper(), "0")

        if st.form_submit_button("Klasifikasikan"):
            vals = [float(v) for v in inputs.values()]
            pred = st.session_state.model.predict([vals])
            label = st.session_state.le.inverse_transform(pred)[0]
            st.success(f"Hasil Prediksi Kualitas Udara: **{label}**")

# ================== RETRAINING ==================
elif menu == "Retraining":
    st.title("⚙️ Re-training Model")
    st.write(
        "Menu ini memungkinkan pengguna mengunggah dataset baru agar model "
        "dilatih ulang (retraining) sehingga dapat beradaptasi dengan data dari kota lain."
    )

    file = st.file_uploader("Upload File CSV", type="csv")
    if file and st.button("Latih Ulang Model"):
        if train_model(pd.read_csv(file)):
            st.success("Model berhasil dilatih ulang menggunakan dataset baru.")
        else:
            st.error("Dataset tidak valid atau kolom 'categori' tidak ditemukan.")
