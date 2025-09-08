import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
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

# --- PERHATIAN: Path sekarang menunjuk ke FOLDER SavedModel ---
MODEL_PATH = Path("models/best_model_tf") 
IMG_SIZE = (256, 256)

# ====================== FUNGSI-FUNGSI UTAMA =========================

@st.cache_resource(show_spinner="Memuat model AI...")
def load_model():
    """
    Memuat model dari format TensorFlow SavedModel. Ini cara paling andal.
    """
    try:
        if not MODEL_PATH.exists():
            st.error(f"❌ Folder model tidak ditemukan di: {MODEL_PATH}", icon="🔥")
            st.warning("Pastikan folder 'best_model_tf' ada di dalam folder 'models'.")
            return None
        
        # tf.keras.models.load_model secara otomatis mengenali format ini
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
        
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {e}", icon="🔥")
        return None

def load_labels():
    """Memuat label kelas secara manual."""
    return [
        'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
        'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 
        'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
        'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
        'Tomato___healthy'
    ]

def preprocess_image(image: Image.Image):
    """Melakukan pra-pemrosesan gambar agar sesuai dengan input model."""
    try:
        img = image.resize(IMG_SIZE).convert("RGB")
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_batch = np.expand_dims(img_array, axis=0)
        return img_batch
    except Exception as e:
        st.error(f"❌ Error saat memproses gambar: {e}", icon="🖼️")
        return None

def predict_image(model, image: Image.Image, class_labels: list):
    """Melakukan prediksi pada gambar."""
    if model is None: return None
    processed_img = preprocess_image(image)
    if processed_img is None: return None

    try:
        predictions = model.predict(processed_img)
        pred_probs = predictions[0]
        predicted_index = np.argmax(pred_probs)
        
        return {
            "label": class_labels[predicted_index],
            "confidence": pred_probs[predicted_index],
            "probabilities": pred_probs
        }
    except Exception as e:
        st.error(f"❌ Gagal melakukan prediksi: {e}", icon="🤯")
        return None

def clean_label(label: str) -> str:
    """Membersihkan nama label untuk ditampilkan."""
    cleaned = label.replace("Tomato___", "").replace("_", " ").strip()
    if cleaned == "healthy": return "Healthy"
    if "Two-spotted spider mite" in cleaned: return "Spider Mites (Two-Spotted)"
    return cleaned.title()

# ====================== DATA DESKRIPSI ==========================
CLASS_DESCRIPTIONS = {
    "Bacterial Spot": "Disebabkan oleh bakteri Xanthomonas. Gejalanya berupa bercak kecil, gelap, dan berair pada daun.",
    "Early Blight": "Disebabkan oleh jamur Alternaria solani. Gejalanya bercak cokelat dengan pola cincin konsentris (target).",
    "Late Blight": "Sangat merusak, disebabkan oleh Phytophthora infestans. Gejalanya berupa bercak hijau gelap berair yang cepat membesar.",
    "Leaf Mold": "Disebabkan oleh jamur Passalora fulva. Gejala khasnya adalah bercak kuning di atas daun dan lapisan jamur di bawahnya.",
    "Septoria Leaf Spot": "Bercak kecil bulat berwarna cokelat dengan pusat abu-abu, disebabkan oleh jamur Septoria lycopersici.",
    "Spider Mites (Two-Spotted)": "Serangan hama tungau laba-laba. Daun menguning dengan bintik-bintik kecil dan jaring halus.",
    "Target Spot": "Bercak cokelat dengan lingkaran konsentris jelas, disebabkan oleh jamur Corynespora cassiicola.",
    "Tomato Yellow Leaf Curl Virus": "Ditransmisikan oleh kutu kebul (whitefly). Daun menguning, menggulung ke atas, dan tanaman kerdil.",
    "Tomato Mosaic Virus": "Menyebabkan pola mosaik (belang-belang) hijau muda dan tua pada daun.",
    "Healthy": "Tanaman sehat dengan daun hijau segar, tanpa tanda-tanda penyakit."
}

# ====================== LOGIKA UTAMA APLIKASI =========================
model = load_model() # Kembali menggunakan fungsi load_model yang sederhana
class_labels = load_labels()

if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None

# --- UI (Tidak ada perubahan) ---
selected = option_menu(
    menu_title=None, options=["Beranda", "Deteksi Penyakit"],
    icons=["house-door-fill", "camera-fill"], orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#fafafa"},
        "nav-link-selected": {"background-color": "#ff4b4b", "color": "white"},
    }
)

if selected == "Beranda":
    st.title("🍅 Klasifikasi Penyakit Daun Tomat")
    st.markdown("Aplikasi ini menggunakan **Convolutional Neural Network (CNN)** untuk mengidentifikasi penyakit pada daun tomat.")
    st.subheader("Jenis Penyakit yang Dapat Dideteksi")
    for raw_label in class_labels:
        cleaned = clean_label(raw_label)
        desc = CLASS_DESCRIPTIONS.get(cleaned, "Deskripsi tidak tersedia.")
        with st.expander(f"**{cleaned}**"):
            st.write(desc)
elif selected == "Deteksi Penyakit":
    st.title("📸 Unggah Gambar untuk Deteksi")
    if model is None:
        st.error("Model tidak dapat digunakan. Harap periksa kembali file model Anda dan restart aplikasi.")
        st.stop()
    source = st.radio("Pilih sumber gambar:", ["Upload File", "Gunakan Kamera"], horizontal=True)
    image_file = None
    if source == "Upload File":
        image_file = st.file_uploader("Pilih gambar daun tomat...", type=["jpg", "jpeg", "png"])
    else:
        image_file = st.camera_input("Arahkan kamera ke daun tomat")
    if image_file:
        img = Image.open(image_file).convert("RGB")
        col1, col2 = st.columns([0.8, 1.2])
        with col1:
            st.image(img, caption="Gambar yang Akan Dianalisis", use_column_width=True)
            if st.button("🔍 Deteksi Sekarang!", type="primary", use_container_width=True):
                with st.spinner("🧠 Menganalisis gambar..."):
                    st.session_state.prediction_result = predict_image(model, img, class_labels)
        with col2:
            st.subheader("Hasil Analisis")
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
                    'Probabilitas': result['probabilities'] * 100
                }).sort_values(by="Probabilitas", ascending=False).reset_index(drop=True)
                st.dataframe(prob_df, use_container_width=True, hide_index=True, column_config={
                    "Probabilitas": st.column_config.ProgressColumn(
                        "Probabilitas (%)", format="%.2f%%", min_value=0, max_value=100,
                    )
                })
            else:
                st.info("Klik tombol 'Deteksi Sekarang!' untuk melihat hasilnya.")
