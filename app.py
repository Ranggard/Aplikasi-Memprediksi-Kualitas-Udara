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
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )

# ================== TRAIN RANDOM FOREST ==================
def train_random_forest(X_train, y_train, n_estimators=100, random_state=42):
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return model

# ================== TRAIN PIPELINE ==================
def train_model(df):
    try:
        # 1. Preprocessing
        X, y, le, df_full = prepare_data(df)

        # 2. Split data
        X_train, X_test, y_train, y_test = split_data(X, y)

        # 3. Train model
        model = train_random_forest(X_train, y_train)

        # 4. Evaluasi
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        st.session_state.update({
            'model': model,
            'le': le,
            'features': X.columns.tolist(),
            'acc': acc,
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

# ================== SIDEBAR ==================
with st.sidebar:
    st.markdown("<br><h2 style='text-align:center;color:#1E88E5;'>Prediksi Kualitas Udara</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;font-size:0.85em;color:gray;'>Random Forest Classifier</p>", unsafe_allow_html=True)

    menu = option_menu(
        menu_title=None,
        options=["Home", "Hasil Latih", "Prediksi Jakarta", "Prediksi Kota Lain", "Retraining"],
        icons=["house-door", "graph-up-arrow", "geo-fill", "map", "cloud-upload"],
        default_index=0
    )

# ================== HOME ==================
if menu == "Home":
    st.title("🏠 Home")
    st.write(
        "Sistem ini menggunakan algoritma **Random Forest** untuk mengklasifikasikan kualitas udara "
        "berdasarkan parameter polutan. Sistem mendukung fitur **re-training** untuk pembaruan model."
    )

    st.subheader("📋 Preview Dataset")
    st.dataframe(st.session_state.df_full.head(5), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.write("Distribusi Kategori ISPU")
        fig, ax = plt.subplots()
        sns.countplot(
            data=st.session_state.df_full,
            x='categori',
            order=KATEGORI_ISPU,
            palette='viridis',
            ax=ax
        )
        plt.xticks(rotation=45)
        st.pyplot(fig)

    with col2:
        st.write("Rata-rata Konsentrasi Polutan")
        st.bar_chart(
            st.session_state.df_full
            .select_dtypes(include=[np.number])
            .mean()
        )

# ================== HASIL LATIH ==================
elif menu == "Hasil Latih":
    st.title("📈 Hasil Latih Model")

    col1, col2 = st.columns(2)
    col1.metric("Akurasi Model", f"{st.session_state.acc*100:.2f}%")
    col2.metric("Jumlah Fitur", len(st.session_state.features))

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(st.session_state.y_test, st.session_state.y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=st.session_state.le.classes_,
        yticklabels=st.session_state.le.classes_,
        ax=ax
    )
    st.pyplot(fig)

    st.subheader("Feature Importance")
    feat_df = pd.DataFrame({
        'Fitur': st.session_state.features,
        'Importance': st.session_state.model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    st.bar_chart(feat_df.set_index('Fitur'))

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
            vals = [float(inputs[f]) for f in st.session_state.features]
            pred = st.session_state.model.predict([vals])
            label = st.session_state.le.inverse_transform(pred)[0]
            st.success(f"Hasil Klasifikasi: **{label}**")

# ================== RETRAINING ==================
elif menu == "Retraining":
    st.title("⚙️ Re-training Model")

    file = st.file_uploader("Upload CSV", type="csv")
    if file and st.button("Mulai Re-training"):
        if train_model(pd.read_csv(file)):
            st.success("Model berhasil diperbarui!")
