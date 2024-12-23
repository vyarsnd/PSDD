import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from streamlit_lottie import st_lottie
import plotly.express as px

def load_lottie_url(url: str):
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()  # Mengembalikan JSON dari URL Lottie
    return None

# Fungsi untuk mengatur tema global
def set_theme():
    st.markdown(
        """
        <style>
        body {
            font-family: 'Arial', sans-serif;
            background-color: #f7f9fc;
        }
        .stApp {
            background-image: url("https://images.pexels.com/photos/3184639/pexels-photo-3184639.jpeg?auto=compress&cs=tinysrgb&dpr=2&h=400");
            background-size: cover;
            background-attachment: fixed;
        }
        h1, h2, h3 {
            color: #5a9;
        }
        </style>
        """, unsafe_allow_html=True
    )


# Fungsi untuk menampilkan header
def display_header():
    st.markdown("<h1 style='text-align: center; color: #5a9; font-family: Arial;'>🌟 Data Science Dashboard 🌟</h1>", unsafe_allow_html=True)
    st.markdown("<hr style='border: 1px solid #5a9;'>", unsafe_allow_html=True)

# Fungsi untuk menampilkan footer
def display_footer():
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style='text-align: center; font-size: small;'>
            <p>© 2024 - Dibuat oleh Navy Arisandi</p>
            <a href="https://github.com/navyarisandi" target="_blank">GitHub</a> |
            <a href="mailto:navy@example.com">Email</a>
        </div>
        """, unsafe_allow_html=True
    )

# Fungsi untuk mengatur background
def set_background():
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url("https://images.pexels.com/photos/3184639/pexels-photo-3184639.jpeg?auto=compress&cs=tinysrgb&dpr=2&h=400");
            background-size: cover;
            background-attachment: fixed;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
# Fungsi untuk progress bar
def show_progress():
    st.info("Sedang dalam proses...")
    for i in range(0, 101, 10):
        st.progress(i)
        st.time.sleep(0.1)

# Fungsi utama
def main():
    set_theme()  # Mengatur tema global
    set_theme()  # Mengatur tema global
    set_background()  # Mengatur background
    display_header()  # Menampilkan header

    st.sidebar.title("📂 Navigasi")
    menu = st.sidebar.radio(
        "Pilih Menu:", ["Beranda", "Upload Dataset", "Preprocessing", "Split Data", "Modeling", "Evaluasi Model", "Prediksi Manual", "Tentang"]
    )

# Contoh URL Lottie
lottie_url = "https://assets1.lottiefiles.com/packages/lf20_jcikwtux.json"  # Ganti dengan URL yang valid
lottie_animation = load_lottie_url(lottie_url)

# Menampilkan animasi di Streamlit
if lottie_animation:
    st_lottie(lottie_animation, height=300)
else:
    st.error("Gagal memuat animasi Lottie. Periksa URL atau koneksi internet Anda.")

st.sidebar.markdown("### 👤 Info Pengguna")
st.sidebar.info("Selamat datang, Navy Arisandi! Pilih menu di atas untuk memulai.")
st.sidebar.markdown("### 🛠️ Bantuan")
st.sidebar.markdown("[Klik di sini](https://streamlit.io/) untuk dokumentasi Streamlit.")

if st.sidebar.button("🌙 Mode Gelap"):
    st.markdown("""
    <style>
    .stApp {background-color: #2E2E2E; color: white;}
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
    .stApp {background-color: #ffffff; color: black;}
    </style>
    """, unsafe_allow_html=True)
with st.spinner("Sedang melakukan preprocessing..."):
    st.progress(50)  # Ini hanya contoh; tingkatkan nilai progres secara dinamis

mode = st.sidebar.selectbox("Pilih Mode Pengguna", ["Beginner", "Advanced"])
if mode == "Beginner":
    st.info("Mode Beginner: Informasi dasar ditampilkan.")
elif mode == "Advanced":
    st.info("Mode Advanced: Menampilkan semua informasi dan pengaturan.")

# Lakukan proses preprocessing di sini
success_animation = "https://assets1.lottiefiles.com/private_files/lf30_cgfdhxgx.json"  # Animasi sukses
success_lottie = load_lottie_url(success_animation)
if success_lottie:
    st_lottie(success_lottie, height=200)
else:
    st.warning("Proses selesai tanpa animasi.")


# Fungsi utama
def main():
    st.sidebar.title("📂 Navigasi")
    menu = st.sidebar.radio(
        "Pilih Menu:", ["Beranda", "Upload Dataset", "Preprocessing", "Split Data", "Modeling", "Evaluasi Model", "Prediksi Manual", "Tentang"]
    )
    if menu == "Beranda":
        st.title("📊 Sistem Informasi Data Science")
        st.image("https://images.pexels.com/photos/669610/pexels-photo-669610.jpeg?auto=compress&cs=tinysrgb&dpr=2&h=400", use_column_width=True)
        lottie_url = "https://assets1.lottiefiles.com/packages/lf20_jcikwtux.json"  # URL animasi Lottie
        animation = load_lottie_url(lottie_url)
        if animation:
            st_lottie(animation, height=300)
        else:
            st.error("Gagal memuat animasi Lottie.")

        st.markdown("""
            ### Selamat Datang
            Sistem ini dirancang untuk membantu analisis data dan perbandingan model machine learning dengan tampilan profesional.
            Gunakan sidebar untuk navigasi fitur.
        """)
        # Menampilkan dataset jika tersedia
        if "data" in st.session_state:
            st.subheader("📋 Dataset")
            st.dataframe(st.session_state["data"], use_container_width=True)

        # Menampilkan hasil evaluasi model jika sudah ada
        if "results_df" in st.session_state:
            st.subheader("📋 Hasil Evaluasi Model Terakhir")
            st.dataframe(st.session_state["results_df"].style.background_gradient(cmap="Blues"), use_container_width=True)

            # Menampilkan data training dan testing
            if "X_train" in st.session_state:
                st.subheader("📊 Data Training dan Testing")
                st.write(f"Jumlah Data Training: {st.session_state['X_train'].shape[0]}")
                st.write(f"Jumlah Data Testing: {st.session_state['X_test'].shape[0]}")

                # Menampilkan seluruh data training dan testing
                st.write("🔹 Data Training:")
                st.dataframe(pd.DataFrame(st.session_state["X_train"], columns=[f"Fitur {i}" for i in range(st.session_state['X_train'].shape[1])]), use_container_width=True)
                st.write("🔹 Data Testing:")
                st.dataframe(pd.DataFrame(st.session_state["X_test"], columns=[f"Fitur {i}" for i in range(st.session_state['X_test'].shape[1])]), use_container_width=True)

            # Menampilkan confusion matrix untuk semua model
            st.subheader("📊 Confusion Matrix Model")
            predictions = st.session_state["predictions"]
            y_test = st.session_state["y_test"]
            for model_name, y_pred in predictions.items():
                st.write(f"**Confusion Matrix: {model_name}**")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm",
                            xticklabels=["Tidak Sakit", "Sakit"], yticklabels=["Tidak Sakit", "Sakit"])
                ax.set_xlabel("Prediksi")
                ax.set_ylabel("Aktual")
                st.pyplot(fig)

            # Menampilkan jumlah data yang sakit dan tidak sakit
            if "data" in st.session_state:
                data = st.session_state["data"]
                if 'target' in data.columns:
                    heart_disease_count = data['target'].value_counts()
                    st.subheader("📊 Jumlah Data yang Mengalami Penyakit Jantung")
                    st.write(f"Jumlah data yang mengalami penyakit jantung (target = 1): {heart_disease_count.get(1, 0)}")
                    st.write(f"Jumlah data yang tidak mengalami penyakit jantung (target = 0): {heart_disease_count.get(0, 0)}")
                    st.write("🔹 Data yang mengidap penyakit jantung:")
                    st.dataframe(data[data['target'] == 1], use_container_width=True)
                    st.write("🔹 Data yang tidak mengidap penyakit jantung:")
                    st.dataframe(data[data['target'] == 0], use_container_width=True)
        else:
            st.warning("Silakan lakukan modeling terlebih dahulu untuk melihat hasilnya.")
        
        st.balloons()
        st.image("https://images.pexels.com/photos/3184639/pexels-photo-3184639.jpeg?auto=compress&cs=tinysrgb&dpr=2&h=400", use_column_width=True)
        modeling_animation_url = "https://assets4.lottiefiles.com/packages/lf20_zrqthn6o.json"
        modeling_animation = load_lottie_url(modeling_animation_url)
        if modeling_animation:
            st_lottie(modeling_animation, height=300)


    elif menu == "Upload Dataset":
        st.header("📁 Upload Dataset Anda")
        uploaded_file = st.file_uploader("Upload File Dataset (CSV)", type=["csv"])

        if uploaded_file:
            data = pd.read_csv(uploaded_file)
            st.success("Dataset berhasil diunggah!")
            st.session_state["data"] = data
            
            col1, col2 = st.columns([3, 2])
            with col1:
                st.subheader("📋 Dataset")
                st.dataframe(data, use_container_width=True)
            with col2:
                st.subheader("📊 Statistik Dataset")
                st.write(data.describe())

            # Statistik dalam kartu
            st.markdown("""
            <div style="padding: 10px; border: 2px solid #5a9; border-radius: 5px; background-color: #f5f5f5;">
                <h3 style="color: #5a9; text-align: center;">📊 Statistik Data</h3>
                <p style="text-align: center;">Dataset memiliki <b>{}</b> baris dan <b>{}</b> fitur.</p>
            </div>
            """.format(data.shape[0], data.shape[1]), unsafe_allow_html=True)

            # Visualisasi awal
            st.subheader("📈 Visualisasi Data")
            fig = px.histogram(data, x=data.columns[0], title="Distribusi {}".format(data.columns[0]))
            st.plotly_chart(fig)

            st.subheader("📋 Dataset")
            st.dataframe(data, use_container_width=True)
            st.image("https://images.pexels.com/photos/3182759/pexels-photo-3182759.jpeg?auto=compress&cs=tinysrgb&dpr=2&h=400", use_column_width=True)
            
            with st.expander("📊 Statistik Dataset"):
                st.write(f"Jumlah baris: {data.shape[0]}")
                st.write(f"Jumlah kolom: {data.shape[1]}")
                st.write("Kolom dataset:")
                st.write(list(data.columns))
                st.write(data.describe())
            st.markdown("""
                        <div style="padding: 10px; border: 2px solid #5a9; border-radius: 5px; background-color: #f5f5f5;">
                            <h3 style="color: #5a9; text-align: center;">📊 Statistik Data</h3>
                            <p style="text-align: center;">Dataset memiliki <b>1000 baris</b> dan <b>10 fitur</b>.</p>
                        </div>
""", unsafe_allow_html=True)

        else:
            st.warning("Silakan unggah file dataset terlebih dahulu.")
            modeling_animation_url = "https://assets4.lottiefiles.com/packages/lf20_zrqthn6o.json"
            modeling_animation = load_lottie_url(modeling_animation_url)
            if modeling_animation:
                st_lottie(modeling_animation, height=300)


    elif menu == "Preprocessing":
        st.header("⚙️ Preprocessing Data")
        if "data" not in st.session_state:
            st.warning("Silakan unggah dataset terlebih dahulu.")
        else:
            data = st.session_state["data"]
            with st.spinner("Sedang melakukan preprocessing..."):
                # Imputasi data
                imputer = SimpleImputer(strategy="median")
                data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

                # Pisahkan fitur dan target
                X = data_imputed.drop("target", axis=1)
                y = data_imputed["target"]

                # Normalisasi dan standardisasi
                normalizer = MinMaxScaler()
                X_normalized = normalizer.fit_transform(X)
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_normalized)

                # Split data
                X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
                st.session_state.update({
                    "X_train": X_train, "X_test": X_test,
                    "y_train": y_train, "y_test": y_test
                })

                st.success("Preprocessing selesai! Data siap digunakan.")

            st.subheader("📋 Hasil Preprocessing")
            tabs = st.tabs(["Imputed Data", "Normalized Data", "Scaled Data"])
            with tabs[0]:
                st.dataframe(data_imputed)
            with tabs[1]:
                st.dataframe(pd.DataFrame(X_normalized, columns=X.columns))
            with tabs[2]:
                st.dataframe(pd.DataFrame(X_scaled, columns=X.columns))
            modeling_animation_url = "https://assets4.lottiefiles.com/packages/lf20_zrqthn6o.json"
            modeling_animation = load_lottie_url(modeling_animation_url)
            if modeling_animation:
                st_lottie(modeling_animation, height=300)


    elif menu == "Split Data":
        st.header("🔄 Data Training dan Testing")
        if "X_train" not in st.session_state:
            st.warning("Silakan lakukan preprocessing terlebih dahulu.")
        else:
            X_train, X_test = st.session_state["X_train"], st.session_state["X_test"]
            st.subheader("📊 Data Training dan Testing")
            st.write(f"Jumlah Data Training: {X_train.shape[0]}")
            st.write(f"Jumlah Data Testing: {X_test.shape[0]}")

            # Menampilkan data training dan testing
            st.write("🔹 Data Training:")
            st.dataframe(pd.DataFrame(X_train, columns=[f"Fitur {i}" for i in range(X_train.shape[1])]), use_container_width=True)
            st.write("🔹 Data Testing:")
            st.dataframe(pd.DataFrame(X_test, columns=[f"Fitur {i}" for i in range(X_test.shape[1])]), use_container_width=True)
            modeling_animation_url = "https://assets4.lottiefiles.com/packages/lf20_zrqthn6o.json"
            modeling_animation = load_lottie_url(modeling_animation_url)
            if modeling_animation:
                st_lottie(modeling_animation, height=300)


    elif menu == "Modeling":
        st.header("🤖 Training Model")
        st.image("https://images.pexels.com/photos/546819/pexels-photo-546819.jpeg?auto=compress&cs=tinysrgb&dpr=2&h=400", use_column_width=True)
        if not all(key in st.session_state for key in ["X_train", "X_test", "y_train", "y_test"]):
            st.warning("Silakan lakukan preprocessing terlebih dahulu.")
        else:
            X_train, X_test = st.session_state["X_train"], st.session_state["X_test"]
            y_train, y_test = st.session_state["y_train"], st.session_state["y_test"]

            st.info("📊 Melatih model...")
            models = {
                "Naive Bayes": GaussianNB(),
                "Random Forest": RandomForestClassifier(random_state=42),
                "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5)
            }

            results = {}
            predictions = {}
            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                results[name] = [accuracy, precision, recall, f1]
                predictions[name] = y_pred

            results_cm = {}
            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                cm = confusion_matrix(y_test, y_pred)
                results_cm[name] = cm
                # Evaluasi
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                results[name] = [accuracy, precision, recall, f1]

            # Simpan confusion matrix di session_state
            st.session_state["confusion_matrices"] = results_cm

            # Menyimpan hasil evaluasi model di session_state untuk ditampilkan di Beranda
            results_df = pd.DataFrame(results, index=["Accuracy", "Precision", "Recall", "F1-score"]).T
            st.session_state["results_df"] = results_df  # Menyimpan hasil ke session state

            st.subheader("📈 Hasil Evaluasi Model")
            st.dataframe(results_df.style.background_gradient(cmap="Blues"), use_container_width=True)

            st.subheader("Visualisasi Performa Model")
            fig, ax = plt.subplots(figsize=(10, 5))
            results_df.plot(kind="bar", ax=ax, colormap="viridis")
            plt.title("Performa Model")
            plt.ylabel("Score")
            plt.xticks(rotation=45)
            st.pyplot(fig)

            st.session_state["models"] = models
            st.session_state["predictions"] = predictions  # Simpan prediksi ke session_state
            st.success("Modeling selesai! Silakan lanjut ke Evaluasi Model.")
            modeling_animation_url = "https://assets4.lottiefiles.com/packages/lf20_zrqthn6o.json"
            modeling_animation = load_lottie_url(modeling_animation_url)
            if modeling_animation:
                st_lottie(modeling_animation, height=300)


    elif menu == "Evaluasi Model":
        st.header("📊 Evaluasi Model dengan Confusion Matrix")
        st.image("https://images.pexels.com/photos/590016/pexels-photo-590016.jpeg?auto=compress&cs=tinysrgb&dpr=2&h=400", use_column_width=True)
        if not all(key in st.session_state for key in ["models", "predictions", "y_test"]):
            st.warning("Silakan lakukan modeling terlebih dahulu.")
        else:
            cm_dict = st.session_state["confusion_matrices"]
            selected_model = st.selectbox("Pilih model untuk Confusion Matrix:", cm_dict.keys())
            cm = cm_dict[selected_model]

            st.subheader(f"Confusion Matrix: {selected_model}")
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm",
                        xticklabels=["Tidak Sakit", "Sakit"], yticklabels=["Tidak Sakit", "Sakit"])
            ax.set_xlabel("Prediksi")
            ax.set_ylabel("Aktual")
            st.pyplot(fig)
            modeling_animation_url = "https://assets4.lottiefiles.com/packages/lf20_zrqthn6o.json"
            modeling_animation = load_lottie_url(modeling_animation_url)
            if modeling_animation:
                st_lottie(modeling_animation, height=300)


    elif menu == "Prediksi Manual":
        st.header("✏️ Prediksi Manual")
        if "X_train" not in st.session_state:
            st.warning("Silakan lakukan preprocessing terlebih dahulu.")
        else:
            X_train = pd.DataFrame(st.session_state["X_train"])
            y_train = st.session_state["y_train"]
            models = st.session_state["models"]
            
            # Input data manual
            manual_input = {}
            for col in X_train.columns:
                manual_input[col] = st.number_input(f"Masukkan nilai untuk {col}:", value=0.0)

            manual_df = pd.DataFrame([manual_input])
            st.write("📋 Input Data Manual")
            st.dataframe(manual_df)

            # Prediksi menggunakan model
            for name, model in models.items():
                prediction = model.predict(manual_df)
                st.write(f"🔹 Prediksi ({name}): {'Sakit' if prediction[0] == 1 else 'Tidak Sakit'}")

    elif menu == "Tentang":
        st.header("📜 Tentang Aplikasi")
        st.markdown("""
            Aplikasi ini dirancang untuk keperluan penelitian akademik terkait analisis data.
            Dibuat oleh 
            NAMA : [NAVY ARISANDI], 
            NIM : [220411100085].
            MATA KULIAH : PROYEK SAINS DATA IF 5B
            UNIVERSITAS TRUNOJOYO MADURA 
        """)
# Footer
    display_footer()

# Menjalankan aplikasi
if __name__ == "__main__":
    main()
