import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import json
from pathlib import Path
import tensorflow as tf
from streamlit_option_menu import option_menu
import base64

# ======================== KONFIGURASI APLIKASI ==========================
st.set_page_config(
    page_title="🍅 Tomato Leaf Disease Classifier",
    page_icon="🍅",
    layout="wide"
)

# Definisikan path untuk model dan label
# Pastikan file-file ini ada di folder yang benar di proyek Streamlit Anda
MODEL_PATH = Path("models/best_model.keras")
LABEL_PATH = Path("models/class_labels.json")
IMG_SIZE = (256, 256)

# ====================== FUNGSI-FUNGSI UTAMA =========================

@st.cache_resource(show_spinner="Memuat model AI...")
def load_model():
    """Memuat model TensorFlow yang telah dilatih. Menggunakan cache agar tidak dimuat ulang."""
    try:
        # Muat model tanpa informasi optimizer untuk inferensi yang lebih cepat
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        return model
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {e}", icon="🔥")
        st.warning("Pastikan file 'best_model.keras' ada di dalam folder 'models'.")
        return None

def load_labels():
    """Memuat label kelas dari file JSON atau menggunakan daftar default."""
    # Prioritaskan memuat dari file JSON untuk sinkronisasi yang akurat
    if LABEL_PATH.exists():
        with open(LABEL_PATH, "r") as f:
            data = json.load(f)
            return data.get("classes", [])
    
    # Daftar fallback jika file JSON tidak ditemukan
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
    """Melakukan prediksi pada gambar yang telah diproses."""
    if model is None:
        st.error("Model tidak tersedia.", icon="🚨")
        return None, None

    processed_img = preprocess_image(image)
    if processed_img is None:
        return None, None

    try:
        with st.spinner("🧠 Menganalisis gambar..."):
            predictions = model.predict(processed_img)
        
        pred_probs = predictions[0]
        predicted_index = np.argmax(pred_probs)
        
        if predicted_index < len(class_labels):
            predicted_label = class_labels[predicted_index]
            confidence = pred_probs[predicted_index]
            return predicted_label, confidence, pred_probs
        else:
            st.warning("Indeks prediksi di luar jangkauan label.", icon="⚠️")
            return "Label tidak diketahui", 0, None

    except Exception as e:
        st.error(f"❌ Gagal melakukan prediksi: {e}", icon="🤯")
        return None, None, None

def clean_label(label: str) -> str:
    """Membersihkan nama label agar lebih mudah dibaca manusia."""
    label = label.replace("Tomato___", "").replace("_", " ").strip()
    return " ".join(part.capitalize() for part in label.split())

# ====================== DATA DESKRIPSI (TETAP SAMA) ==========================
CLASS_DESCRIPTIONS = {
    "Bacterial Spot": "Disebabkan oleh bakteri Xanthomonas, gejalanya berupa bercak kecil gelap pada daun.",
    "Early Blight": "Disebabkan oleh jamur Alternaria solani, gejalanya bercak cokelat dengan pola cincin konsentris.",
    "Late Blight": "Sangat merusak, disebabkan oleh Phytophthora infestans. Bercak hijau gelap berair yang cepat meluas.",
    "Leaf Mold": "Disebabkan oleh jamur Passalora fulva, dengan bercak kuning di atas daun dan lapisan beludru di bawahnya.",
    "Septoria Leaf Spot": "Bercak kecil bulat berwarna cokelat dengan pusat abu-abu, disebabkan oleh jamur Septoria lycopersici.",
    "Spider Mites Two-Spotted Spider Mite": "Serangan hama tungau laba-laba. Daun menguning dengan bintik-bintik dan jaring halus.",
    "Target Spot": "Bercak cokelat dengan lingkaran konsentris mirip sasaran tembak, disebabkan oleh jamur Corynespora.",
    "Tomato Yellow Leaf Curl Virus": "Ditransmisikan oleh kutu kebul. Daun menguning, menggulung ke atas, dan tanaman kerdil.",
    "Tomato Mosaic Virus": "Menyebabkan pola mosaik belang hijau muda dan tua pada daun, serta pertumbuhan abnormal.",
    "Healthy": "Tanaman sehat dengan daun hijau segar, bebas dari bercak atau kelainan bentuk."
}

# ====================== MEMUAT MODEL & LABEL =========================
model = load_model()
class_labels = load_labels()

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
    st.markdown("Aplikasi ini menggunakan **Convolutional Neural Network (CNN)** untuk mengidentifikasi 9 jenis penyakit umum pada daun tomat dan membedakannya dari daun yang sehat.")
    
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
        st.error("Tidak dapat melanjutkan karena model gagal dimuat.")
        st.stop()
        
    source = st.radio("Pilih sumber gambar:", ["Upload File", "Gunakan Kamera"], horizontal=True)
    
    image_file = None
    if source == "Upload File":
        image_file = st.file_uploader("Pilih gambar daun tomat...", type=["jpg", "jpeg", "png"])
    elif source == "Gunakan Kamera":
        image_file = st.camera_input("Arahkan kamera ke daun tomat")

    if image_file:
        try:
            img = Image.open(image_file).convert("RGB")
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(img, caption="Gambar Anda", use_column_width=True)

            with col2:
                if st.button("🔍 Deteksi Sekarang!", type="primary", use_container_width=True):
                    label, confidence, all_probs = predict_image(model, img, class_labels)

                    if label and confidence is not None:
                        cleaned_label = clean_label(label)
                        st.success(f"**Hasil Deteksi: {cleaned_label}**")
                        st.info(f"**Tingkat Keyakinan: {confidence:.2%}**")
                        
                        st.subheader("Deskripsi")
                        desc = CLASS_DESCRIPTIONS.get(cleaned_label, "Deskripsi tidak tersedia.")
                        st.write(desc)

                        st.subheader("Probabilitas Semua Kelas")
                        prob_df = pd.DataFrame({
                            'Kelas': [clean_label(cls) for cls in class_labels],
                            'Probabilitas': all_probs
                        }).sort_values(by="Probabilitas", ascending=False).reset_index(drop=True)
                        
                        st.dataframe(prob_df, use_container_width=True)

        except Exception as e:
            st.error(f"Gagal memproses file gambar: {e}")
