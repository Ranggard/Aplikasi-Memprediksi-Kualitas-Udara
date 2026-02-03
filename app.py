import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import joblib

# =========================
# KONFIGURASI HALAMAN
# =========================
st.set_page_config(page_title="Klasifikasi Pencemaran Udara", layout="wide")

st.title("🌫️ Sistem Klasifikasi Pencemaran Udara")
st.caption("Random Forest dengan pendekatan Re-Training (Jakarta & Kota Lain)")

# =========================
# FUNGSI UTILITAS
# =========================
def load_data(path):
    return pd.read_csv(path)


def preprocess_data(df):
    df = df.copy()
    df = df.dropna()

    category_map = {
        'BAIK': 1,
        'SEDANG': 2,
        'TIDAK SEHAT': 3,
        'SANGAT TIDAK SEHAT': 4,
        'BERBAHAYA': 5
    }

    if 'categori' in df.columns:
        df['categori'] = df['categori'].map(category_map)

    features = ['pm10', 'pm25', 'so2', 'co', 'o3']
    df = df[df[features].apply(lambda x: (x >= 0).all(), axis=1)]

    return df, features


def train_model(X, y):
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )
    model.fit(X, y)
    return model


# =========================
# SIDEBAR MENU
# =========================
menu = st.sidebar.selectbox(
    "Menu",
    ["Dashboard", "Prediksi Kota Jakarta", "Prediksi Kota Lain", "Upload Dataset & Re-Training"]
)

# =========================
# LOAD DATASET JAKARTA
# =========================
DATA_URL = "https://raw.githubusercontent.com/Ranggard/Dataset/main/Quality_Air_Jakarta.csv"

df_raw = load_data(DATA_URL)
df, FEATURES = preprocess_data(df_raw)

X = df[FEATURES]
y = df['categori']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = train_model(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# =========================
# DASHBOARD
# =========================
if menu == "Dashboard":
    st.subheader("📊 Dashboard Model")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Akurasi Model (Jakarta)", f"{acc*100:.2f}%")

    with col2:
        st.write("Jumlah Data:")
        st.write(df.shape)

    st.subheader("Distribusi Kategori ISPU")
    fig, ax = plt.subplots()
    df['categori'].value_counts().sort_index().plot(kind='bar', ax=ax)
    st.pyplot(fig)

# =========================
# PREDIKSI JAKARTA
# =========================
elif menu == "Prediksi Kota Jakarta":
    st.subheader("🏙️ Prediksi Kualitas Udara Kota Jakarta")

    input_data = {}
    cols = st.columns(len(FEATURES))

    for i, feature in enumerate(FEATURES):
        input_data[feature] = cols[i].number_input(feature.upper(), min_value=0.0)

    if st.button("Prediksi"):
        input_df = pd.DataFrame([input_data])
        result = model.predict(input_df)[0]

        st.success(f"Hasil Kategori ISPU: {result}")

# =========================
# PREDIKSI KOTA LAIN
# =========================
elif menu == "Prediksi Kota Lain":
    st.subheader("🌆 Prediksi Kota Lain (Tanpa Re-Training)")

    st.warning("Akurasi dapat menurun karena model hanya dilatih menggunakan data Jakarta")

    input_data = {}
    cols = st.columns(len(FEATURES))

    for i, feature in enumerate(FEATURES):
        input_data[feature] = cols[i].number_input(feature.upper(), min_value=0.0)

    if st.button("Prediksi Kota Lain"):
        input_df = pd.DataFrame([input_data])
        result = model.predict(input_df)[0]

        st.info(f"Hasil Prediksi ISPU Kota Lain: {result}")

# =========================
# RE-TRAINING
# =========================
elif menu == "Upload Dataset & Re-Training":
    st.subheader("🔁 Upload Dataset & Re-Training Model")

    uploaded_file = st.file_uploader("Upload dataset kota lain (.csv)", type=["csv"])

    if uploaded_file:
        df_new = pd.read_csv(uploaded_file)
        df_new, _ = preprocess_data(df_new)

        X_new = df_new[FEATURES]
        y_new = df_new['categori']

        model_retrain = train_model(X_new, y_new)
        y_new_pred = model_retrain.predict(X_new)
        acc_new = accuracy_score(y_new, y_new_pred)

        st.success(f"Re-Training selesai | Akurasi dataset baru: {acc_new*100:.2f}%")

        fig, ax = plt.subplots()
        cm = confusion_matrix(y_new, y_new_pred)
        ConfusionMatrixDisplay(cm).plot(ax=ax)
        st.pyplot(fig)
