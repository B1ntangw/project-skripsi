import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import json
from pathlib import Path
import tensorflow as tf
from streamlit_option_menu import option_menu

# ======================== KONFIGURASI APLIKASI ==========================
st.set_page_config(
    page_title="🍅 Tomato Leaf Disease Classifier",
    page_icon="🍅",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Definisikan path untuk model dan label
# Pastikan file-file ini ada di folder yang benar di proyek Streamlit Anda
MODEL_PATH = Path("models/best_model.keras")
LABEL_PATH = Path("models/class_labels.json") # Opsional, jika Anda punya file JSON
IMG_SIZE = (256, 256)

# ====================== FUNGSI-FUNGSI UTAMA =========================

@st.cache_resource(show_spinner="Memuat model AI...")
def load_model():
    """Memuat model TensorFlow yang telah dilatih. Menggunakan cache agar tidak dimuat ulang."""
    try:
        if not MODEL_PATH.exists():
            st.error(f"❌ File model tidak ditemukan di: {MODEL_PATH}", icon="🔥")
            st.warning("Pastikan file 'best_model.keras' ada di dalam folder 'models'.")
            return None
        # Muat model tanpa informasi optimizer untuk inferensi yang lebih cepat
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        return model
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {e}", icon="🔥")
        return None

def load_labels():
    """Memuat label kelas dari file JSON atau menggunakan daftar default."""
    if LABEL_PATH.exists():
        try:
            with open(LABEL_PATH, "r") as f:
                data = json.load(f)
            return data.get("classes", [])
        except Exception as e:
            st.warning(f"Gagal membaca file label JSON: {e}. Menggunakan daftar default.")
    
    # PERHATIAN: Urutan ini harus sama persis dengan urutan saat training model
    return [
        "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
        "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
        "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus",
        "Tomato___healthy"
    ]

def preprocess_image(image: Image.Image):
    """Melakukan pra-pemrosesan gambar agar sesuai dengan input model."""
    try:
        img = image.resize(IMG_SIZE).convert("RGB")
        img_array = np.array(img, dtype=np.float32) / 255.0
        # Tambahkan dimensi batch -> dari (256, 256, 3) menjadi (1, 256, 256, 3)
        img_batch = np.expand_dims(img_array, axis=0)
        return img_batch
    except Exception as e:
        st.error(f"❌ Error saat memproses gambar: {e}", icon="🖼️")
        return None

def predict_image(model, image: Image.Image, class_labels: list):
    """Melakukan prediksi pada gambar dan mengembalikan hasil lengkap."""
    if model is None:
        st.error("Model tidak tersedia untuk prediksi.", icon="🚨")
        return None

    processed_img = preprocess_image(image)
    if processed_img is None:
        return None

    try:
        predictions = model.predict(processed_img)
        pred_probs = predictions[0]
        predicted_index = np.argmax(pred_probs)
        
        if predicted_index < len(class_labels):
            predicted_label = class_labels[predicted_index]
            confidence = pred_probs[predicted_index]
            return {
                "label": predicted_label,
                "confidence": confidence,
                "probabilities": pred_probs
            }
        else:
            st.warning("Indeks prediksi di luar jangkauan label.", icon="⚠️")
            return None
    except Exception as e:
        st.error(f"❌ Gagal melakukan prediksi: {e}", icon="🤯")
        return None

def clean_label(label: str) -> str:
    """Membersihkan nama label agar lebih mudah dibaca manusia."""
    return label.replace("Tomato___", "").replace("_", " ").strip().title()

# ====================== DATA DESKRIPSI (TETAP SAMA) ==========================
CLASS_DESCRIPTIONS = {
    "Bacterial Spot": "Disebabkan oleh bakteri Xanthomonas. Gejalanya berupa bercak kecil, gelap, dan berair pada daun yang seringkali memiliki lingkaran kuning di sekelilingnya.",
    "Early Blight": "Disebabkan oleh jamur Alternaria solani. Gejalanya adalah bercak cokelat dengan pola cincin konsentris (seperti target tembak), biasanya dimulai dari daun bagian bawah.",
    "Late Blight": "Sangat merusak, disebabkan oleh oomycete Phytophthora infestans. Gejalanya berupa bercak hijau gelap berair yang cepat membesar menjadi cokelat dan dapat menyebar ke batang dan buah.",
    "Leaf Mold": "Disebabkan oleh jamur Passalora fulva. Gejala khasnya adalah bercak kuning di permukaan atas daun, sementara di bagian bawahnya terdapat lapisan jamur berwarna hijau zaitun hingga cokelat.",
    "Septoria Leaf Spot": "Bercak kecil bulat berwarna cokelat dengan pusat abu-abu atau cokelat muda, disebabkan oleh jamur Septoria lycopersici. Biasanya dimulai dari daun bawah dan menyebar ke atas.",
    "Spider Mites Two-Spotted Spider Mite": "Ini adalah serangan hama tungau laba-laba. Daun tampak menguning dengan bintik-bintik kecil, dan jika parah, akan terlihat jaring halus di bawah daun.",
    "Target Spot": "Bercak cokelat dengan lingkaran konsentris yang jelas mirip sasaran tembak, disebabkan oleh jamur Corynespora cassiicola.",
    "Tomato Yellow Leaf Curl Virus": "Ditransmisikan oleh kutu kebul (whitefly). Gejalanya meliputi daun yang menguning, menggulung ke atas, menjadi lebih kecil, dan pertumbuhan tanaman menjadi kerdil.",
    "Tomato Mosaic Virus": "Menyebabkan pola mosaik (belang-belang) antara area hijau muda dan hijau tua pada daun. Daun juga bisa tampak keriput atau pertumbuhannya tidak normal.",
    "Healthy": "Tanaman sehat dengan daun berwarna hijau segar dan merata, tanpa ada tanda-tanda bercak, perubahan warna, atau kelainan bentuk."
}

# ====================== MEMUAT MODEL & LABEL =========================
model = load_model()
class_labels = load_labels()

# Inisialisasi session state untuk menyimpan hasil
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None

# ========================== UI STREAMLIT ============================

# --- NAVIGASI ---
selected = option_menu(
    menu_title=None,
    options=["Beranda", "Deteksi Penyakit"],
    icons=["house-door-fill", "camera-fill"],
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#fafafa"},
        "nav-link-selected": {"background-color": "#ff4b4b", "color": "white"},
    }
)

# --- HALAMAN BERANDA ---
if selected == "Beranda":
    st.title("🍅 Klasifikasi Penyakit Daun Tomat")
    st.markdown("Aplikasi ini menggunakan **Convolutional Neural Network (CNN)** untuk mengidentifikasi 9 jenis penyakit umum dan hama pada daun tomat, serta membedakannya dari daun yang sehat.")
    
    st.subheader("Jenis Penyakit yang Dapat Dideteksi")
    for raw_label in class_labels:
        cleaned = clean_label(raw_label)
        desc = CLASS_DESCRIPTIONS.get(cleaned, "Deskripsi tidak tersedia.")
        with st.expander(f"**{cleaned}**"):
            st.write(desc)

# --- HALAMAN DETEKSI ---
elif selected == "Deteksi Penyakit":
    st.title("📸 Unggah Gambar untuk Deteksi")
    
    if model is None:
        st.error("Tidak dapat melanjutkan karena model gagal dimuat. Silakan periksa log server.")
        st.stop()
        
    source = st.radio("Pilih sumber gambar:", ["Upload File", "Gunakan Kamera"], horizontal=True, key="image_source")
    
    image_file = None
    if source == "Upload File":
        image_file = st.file_uploader("Pilih gambar daun tomat...", type=["jpg", "jpeg", "png"], key="file_uploader")
    elif source == "Gunakan Kamera":
        image_file = st.camera_input("Arahkan kamera ke daun tomat", key="camera_input")

    if image_file:
        try:
            img = Image.open(image_file).convert("RGB")
            
            col1, col2 = st.columns([0.8, 1.2]) # Beri lebih banyak ruang untuk hasil
            with col1:
                st.image(img, caption="Gambar yang Akan Dianalisis", use_column_width=True)
                
                # Tombol deteksi
                if st.button("🔍 Deteksi Sekarang!", type="primary", use_container_width=True):
                    with st.spinner("🧠 Menganalisis gambar... Mohon tunggu."):
                        # Lakukan prediksi dan simpan hasilnya di session_state
                        st.session_state.prediction_result = predict_image(model, img, class_labels)

            with col2:
                st.subheader("Hasil Analisis")
                # Tampilkan hasil dari session_state jika ada
                if st.session_state.prediction_result:
                    result = st.session_state.prediction_result
                    cleaned_label = clean_label(result['label'])
                    
                    st.success(f"**Hasil Deteksi: {cleaned_label}**")
                    st.info(f"**Tingkat Keyakinan: {result['confidence']:.2%}**")
                    
                    st.markdown("---")
                    
                    st.write("**Deskripsi:**")
                    desc = CLASS_DESCRIPTIONS.get(cleaned_label, "Deskripsi tidak tersedia.")
                    st.write(desc)
                    
                    st.markdown("---")

                    st.write("**Probabilitas Semua Kelas:**")
                    prob_df = pd.DataFrame({
                        'Kelas': [clean_label(cls) for cls in class_labels],
                        'Probabilitas': result['probabilities']
                    }).sort_values(by="Probabilitas", ascending=False).reset_index(drop=True)
                    
                    st.dataframe(prob_df, 
                                 use_container_width=True,
                                 hide_index=True,
                                 column_config={
                                     "Probabilitas": st.column_config.ProgressColumn(
                                         "Probabilitas",
                                         format="%.2f%%",
                                         min_value=0,
                                         max_value=1,
                                     )
                                 })
                else:
                    st.info("Unggah gambar dan klik tombol 'Deteksi Sekarang!' untuk melihat hasilnya.")

        except Exception as e:
            st.error(f"Gagal memproses file gambar: {e}")
