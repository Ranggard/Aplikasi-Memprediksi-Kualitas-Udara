# ================== LIBRARY ==================
import streamlit as st
import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    precision_score, recall_score, f1_score,
    classification_report
)
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_option_menu import option_menu

# ================== KONFIGURASI ==================
st.set_page_config(page_title="Prediksi Kualitas Udara", layout="wide")

KATEGORI_ISPU = ['BAIK', 'SEDANG', 'TIDAK SEHAT', 'SANGAT TIDAK SEHAT', 'BERBAHAYA']

# ================== SESSION STATE ==================
if 'model' not in st.session_state:
    st.session_state.update({
        'model': None,
        'le': None,
        'features': [],
        'acc': 0,
        'precision': 0,
        'recall': 0,
        'f1': 0,
        'train_time': 0,
        'df_full': None,
        'y_test': None,
        'y_pred': None
    })

# ================== PREPROCESSING ==================
def prepare_data(df):
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

    return X, y, le, df

# ================== SPLIT DATA ==================
def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# ================== TRAIN RANDOM FOREST ==================
def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

# ================== TRAIN PIPELINE ==================
def train_model(df):
    try:
        X, y, le, df_full = prepare_data(df)
        X_train, X_test, y_train, y_test = split_data(X, y)

        start_time = time.time()
        model = train_random_forest(X_train, y_train)
        train_time = time.time() - start_time

        y_pred = model.predict(X_test)

        st.session_state.update({
            'model': model,
            'le': le,
            'features': X.columns.tolist(),
            'acc': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted'),
            'train_time': train_time,
            'df_full': df_full,
            'y_test': y_test,
            'y_pred': y_pred
        })
        return True
    except Exception as e:
        st.error(f"Training gagal: {e}")
        return False

# ================== LOAD DEFAULT DATA ==================
if st.session_state.model is None:
    url = "https://raw.githubusercontent.com/Ranggard/Dataset/main/Quality_Air_Jakarta.csv"
    train_model(pd.read_csv(url))

# ================== FUNGSI KEPUTUSAN ==================
def keputusan_udara(kategori, kota):
    keputusan = {
        "BAIK": "aman untuk aktivitas luar ruangan dan mendukung kegiatan ekonomi.",
        "SEDANG": "masih dapat dilakukan dengan kewaspadaan.",
        "TIDAK SEHAT": "disarankan mengurangi aktivitas luar ruangan.",
        "SANGAT TIDAK SEHAT": "aktivitas luar ruangan sebaiknya dibatasi.",
        "BERBAHAYA": "aktivitas luar ruangan sebaiknya dihentikan."
    }
    return f"Kualitas udara di Kota {kota} tergolong {kategori}, sehingga {keputusan[kategori]}"

# ================== SIDEBAR ==================
with st.sidebar:
    st.markdown("<h2 style='text-align:center;color:#1E88E5;'>Prediksi Kualitas Udara</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:gray;'>Random Forest Classifier</p>", unsafe_allow_html=True)

    menu = option_menu(
        None,
        ["Home", "Hasil Latih", "Prediksi Jakarta", "Prediksi Kota Lain", "Retraining"],
        icons=["house", "graph-up", "geo", "map", "cloud-upload"]
    )

# ================== HOME ==================
if menu == "Home":
    st.title("🏠 Home")
    st.write("Aplikasi ini merupakan sistem pendukung keputusan kualitas udara berbasis Random Forest.")
    st.dataframe(st.session_state.df_full.head(5), use_container_width=True)

# ================== HASIL LATIH ==================
elif menu == "Hasil Latih":
    st.title("📈 Hasil Latih Model")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", f"{st.session_state.acc*100:.2f}%")
    c2.metric("Precision", f"{st.session_state.precision*100:.2f}%")
    c3.metric("Recall", f"{st.session_state.recall*100:.2f}%")
    c4.metric("F1-Score", f"{st.session_state.f1*100:.2f}%")
    c5.metric("Waktu Latih (detik)", f"{st.session_state.train_time:.2f}")

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(st.session_state.y_test, st.session_state.y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=st.session_state.le.classes_,
                yticklabels=st.session_state.le.classes_, ax=ax)
    st.pyplot(fig)

    st.subheader("Feature Importance")
    feat_df = pd.DataFrame({
        "Fitur": st.session_state.features,
        "Importance": st.session_state.model.feature_importances_
    }).sort_values(by="Importance", ascending=False)
    st.bar_chart(feat_df.set_index("Fitur"))

# ================== PREDIKSI JAKARTA ==================
elif menu == "Prediksi Jakarta":
    st.title("🔍 Prediksi Kualitas Udara Jakarta")

    with st.form("jakarta"):
        inputs = {}
        cols = st.columns(len(st.session_state.features))
        for i, f in enumerate(st.session_state.features):
            with cols[i]:
                inputs[f] = st.text_input(f.upper(), "0")

        if st.form_submit_button("Klasifikasikan"):
            vals = [float(inputs[f]) for f in st.session_state.features]
            label = st.session_state.le.inverse_transform(
                st.session_state.model.predict([vals])
            )[0]
            st.success(f"Hasil Klasifikasi: **{label}**")
            st.info(keputusan_udara(label, "Jakarta"))

# ================== PREDIKSI KOTA LAIN ==================
elif menu == "Prediksi Kota Lain":
    st.title("🔍 Prediksi Kualitas Udara Kota Lain")
    nama_kota = st.text_input("Nama Kota")

    with st.form("kota_lain"):
        inputs = {}
        cols = st.columns(len(st.session_state.features))
        for i, f in enumerate(st.session_state.features):
            with cols[i]:
                inputs[f] = st.text_input(f.upper(), "0")

        if st.form_submit_button("Klasifikasikan"):
            if nama_kota.strip() == "":
                st.error("Nama kota tidak boleh kosong.")
            else:
                vals = [float(inputs[f]) for f in st.session_state.features]
                label = st.session_state.le.inverse_transform(
                    st.session_state.model.predict([vals])
                )[0]
                st.success(f"Hasil Klasifikasi Kota {nama_kota}: **{label}**")
                st.info(keputusan_udara(label, nama_kota))
                st.warning(
                    "Catatan: Model dilatih menggunakan dataset Kota Jakarta, "
                    "sehingga akurasi prediksi untuk kota lain dapat berkurang."
                )

# ================== RETRAINING ==================
elif menu == "Retraining":
    st.title("⚙️ Re-training Model")
    file = st.file_uploader("Upload CSV", type="csv")
    if file and st.button("Mulai Re-training"):
        if train_model(pd.read_csv(file)):
            st.success("Model berhasil diperbarui!")
