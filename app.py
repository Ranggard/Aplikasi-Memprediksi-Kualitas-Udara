# =======================
# LIBRARY
# =======================
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

# =======================
# KONFIGURASI
# =======================
st.set_page_config(page_title="Prediksi Kualitas Udara", layout="wide")

RULES = {
    'pm10': (0, 500), 'pm25': (0, 500), 'so2': (0, 800),
    'co': (0, 100), 'o3': (0, 600), 'no2': (0, 800)
}

KATEGORI_ISPU = ['BAIK', 'SEDANG', 'TIDAK SEHAT', 'SANGAT TIDAK SEHAT', 'BERBAHAYA']

# =======================
# SESSION STATE
# =======================
if 'model' not in st.session_state:
    st.session_state.update({
        'model': None,
        'le': None,
        'features': [],
        'n_features': 0,
        'n_train': 0,
        'n_test': 0,
        'acc': 0,
        'df_full': None
    })

# =======================
# FUNGSI TRAINING
# =======================
def train_model(df):
    try:
        df['categori'] = df['categori'].str.upper().str.strip()

        drop_cols = ['tanggal', 'stasiun', 'critical', 'location']
        df_clean = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
        df_clean = df_clean.dropna()

        le = LabelEncoder()
        df_clean['categori'] = le.fit_transform(df_clean['categori'])

        X = df_clean.select_dtypes(include=[np.number]).drop(columns=['categori'], errors='ignore')
        y = df_clean['categori']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
        model.fit(X_train, y_train)

        acc = accuracy_score(y_test, model.predict(X_test))

        st.session_state.update({
            'model': model,
            'le': le,
            'features': X.columns.tolist(),
            'n_features': X.shape[1],
            'n_train': X_train.shape[0],
            'n_test': X_test.shape[0],
            'acc': acc,
            'df_full': df_clean
        })
        return True
    except Exception as e:
        st.error(f"Error training: {e}")
        return False

# =======================
# LOAD DATA DEFAULT
# =======================
if st.session_state.model is None:
    url = "https://raw.githubusercontent.com/Ranggard/Dataset/main/Quality_Air_Jakarta.csv"
    try:
        train_model(pd.read_csv(url))
    except:
        st.error("Gagal memuat dataset default.")

# =======================
# SIDEBAR
# =======================
with st.sidebar:
    st.markdown("## 🌫️ Prediksi Kualitas Udara")
    st.caption("Random Forest Classifier")

    menu = option_menu(
        None,
        ["Home", "Hasil Latih", "Prediksi Jakarta", "Prediksi Kota Lain", "Retraining"],
        icons=["house", "graph-up", "geo", "map", "arrow-repeat"],
        default_index=0
    )

# =======================
# HOME
# =======================
if menu == "Home":
    st.title("🏠 Home")

    st.markdown("""
    Aplikasi ini menggunakan **Random Forest Classifier** untuk memprediksi
    kategori kualitas udara berdasarkan parameter polutan.

    Model mendukung **retraining dinamis**, sehingga dapat digunakan
    untuk kota lain dengan menambahkan dataset baru.
    """)

    st.divider()
    st.subheader("📋 Contoh Data")
    st.dataframe(st.session_state.df_full.head(), use_container_width=True)

# =======================
# HASIL LATIH
# =======================
elif menu == "Hasil Latih":
    st.title("📈 Hasil Latih Model")

    col1, col2, col3 = st.columns(3)
    col1.metric("Akurasi Model", f"{st.session_state.acc*100:.2f}%")
    col2.metric("Jumlah Fitur", st.session_state.n_features)
    col3.metric("Data Latih / Uji", f"{st.session_state.n_train} / {st.session_state.n_test}")

    st.divider()

    # Feature Importance
    feat_df = pd.DataFrame({
        "Fitur": st.session_state.features,
        "Importance": st.session_state.model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    st.subheader("🔎 Feature Importance")
    st.bar_chart(feat_df.set_index("Fitur"))

    st.divider()

    # Confusion Matrix
    st.subheader("📊 Confusion Matrix")

    X = st.session_state.df_full[st.session_state.features]
    y_true = st.session_state.df_full['categori']
    y_pred = st.session_state.model.predict(X)

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=st.session_state.le.classes_,
        yticklabels=st.session_state.le.classes_
    )
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Aktual")
    st.pyplot(fig)

    st.caption(
        "Mayoritas nilai berada pada diagonal utama, "
        "menunjukkan performa klasifikasi yang baik."
    )

# =======================
# PREDIKSI
# =======================
elif menu in ["Prediksi Jakarta", "Prediksi Kota Lain"]:
    st.title(f"🔍 {menu}")

    with st.form("form_prediksi"):
        cols = st.columns(len(st.session_state.features))
        input_data = []

        for i, f in enumerate(st.session_state.features):
            with cols[i]:
                val = st.text_input(f.upper(), "0")
                input_data.append(val)

        if st.form_submit_button("Klasifikasikan"):
            try:
                input_data = [float(v) for v in input_data]
                pred = st.session_state.model.predict([input_data])
                label = st.session_state.le.inverse_transform(pred)[0]
                st.success(f"### Hasil Klasifikasi: **{label}**")
            except:
                st.error("Input harus berupa angka valid.")

# =======================
# RETRAINING
# =======================
elif menu == "Retraining":
    st.title("⚙️ Retraining Model")

    file = st.file_uploader("Upload file CSV", type="csv")
    if file and st.button("Latih Ulang"):
        if train_model(pd.read_csv(file)):
            st.success("Model berhasil diperbarui!")
        else:
            st.error("Format dataset tidak sesuai.")
